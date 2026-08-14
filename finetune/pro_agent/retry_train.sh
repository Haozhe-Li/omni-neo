#!/bin/bash
# Retry the v4 SFT run until W&B Serverless Training starts executing jobs again.
#
# Why this exists: on 2026-08-13 the backend accepted jobs and then never ran
# them. Registration succeeded, the data artifact uploaded, "Starting SFT
# training job..." printed, and nothing further happened — no steps, no error,
# and only the step-0 checkpoint every model gets at registration. Three runs
# hung identically: the full 157-row v4 (2h15m), a 4-row smoke of the same data
# (25m), and a 4-row control built from v3's own already-trained file (22m).
# The control is what rules out the data: v3's rows trained fine on 2026-08-12.
#
# The loop probes with 4 rows before committing to a full run. A hung probe
# costs 15 minutes; a hung full run costs an hour or more, and that asymmetry
# is the whole design — the v2 episode burned several hours by retrying the
# expensive thing blind.
set -u
cd "$(dirname "$0")/../.."
PY=venv/bin/python3.12
SCRATCH="${SCRATCH:-${TMPDIR:-/tmp}/omni-train-logs}"
mkdir -p "$SCRATCH"
DATA=finetune/pro_agent/dataset
PROBE=$DATA/sft_probe.jsonl
# Overridable so the next version does not need this file edited. Defaults kept
# at v4 so an old invocation still means what it meant.
FULL="${FULL:-$DATA/sft_train_v4.jsonl}"
RUN_PREFIX="${RUN_PREFIX:-omni-pro-v4}"
EPOCHS="${EPOCHS:-6}"
ATTEMPTS=${ATTEMPTS:-10}
PROBE_WAIT=${PROBE_WAIT:-30}      # x30s = 15 min
FULL_WAIT=${FULL_WAIT:-200}       # x30s = 100 min
COOLDOWN=${COOLDOWN:-1500}        # 25 min between attempts

# 4 rows: the three shortest plus the longest, so a pass covers the worst case.
$PY - <<PYEOF
import json
rows=[json.loads(l) for l in open("$FULL")]
rows.sort(key=lambda r: len(json.dumps(r,ensure_ascii=False)))
with open("$PROBE","w",encoding="utf-8") as f:
    for r in rows[:3]+[rows[-1]]: f.write(json.dumps(r,ensure_ascii=False)+"\n")
PYEOF

# Waits for `trained in` (train.py prints it only after train_sft returns).
# Returns 0 on success, 1 on hang or error; kills the child either way.
wait_for() {
  local log="$1" ticks="$2" pid="$3" i
  for ((i=0; i<ticks; i++)); do
    grep -q "trained in" "$log" 2>/dev/null && return 0
    grep -qE "Traceback|Error code" "$log" 2>/dev/null && return 1
    kill -0 "$pid" 2>/dev/null || return 1
    sleep 30
  done
  return 1
}

for ((a=1; a<=ATTEMPTS; a++)); do
  ts=$(date +%m%d-%H%M)
  echo "[$(date +%H:%M)] attempt $a/$ATTEMPTS — probing with 4 rows"
  $PY -u finetune/pro_agent/train.py --file "$PROBE" --epochs 1 \
      --name "probe-$ts" > "$SCRATCH/probe_$ts.log" 2>&1 &
  ppid=$!
  if wait_for "$SCRATCH/probe_$ts.log" "$PROBE_WAIT" "$ppid"; then
    echo "[$(date +%H:%M)] probe OK — backend is executing jobs; starting full v4"
    kill $ppid 2>/dev/null
    $PY -u finetune/pro_agent/train.py --file "$FULL" --epochs "$EPOCHS" \
        --name "$RUN_PREFIX-$ts" > "$SCRATCH/train_v4_$ts.log" 2>&1 &
    fpid=$!
    if wait_for "$SCRATCH/train_v4_$ts.log" "$FULL_WAIT" "$fpid"; then
      echo "[$(date +%H:%M)] TRAINED — $RUN_PREFIX-$ts"
      grep -E "trained in|inference name|Artifact URL" "$SCRATCH/train_v4_$ts.log"
      exit 0
    fi
    echo "[$(date +%H:%M)] full run hung after a passing probe — backend degraded mid-run"
    kill $fpid 2>/dev/null
  else
    echo "[$(date +%H:%M)] probe hung/failed — backend still down"
    kill $ppid 2>/dev/null
  fi
  pkill -f "[t]rain.py" 2>/dev/null
  [ "$a" -lt "$ATTEMPTS" ] && { echo "  cooling down ${COOLDOWN}s"; sleep "$COOLDOWN"; }
done
echo "[$(date +%H:%M)] gave up after $ATTEMPTS attempts — backend never recovered"
exit 1
