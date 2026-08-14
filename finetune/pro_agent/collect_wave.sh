#!/bin/bash
# Collect a set of queries with N processes in flight, then merge into the
# shared trace files.
#
#   ./collect_wave.sh <log-dir> <concurrency> <query-id> [query-id ...]
#
# One process per query, each with its own `--out`. That separation is not
# stylistic: `collect.py` appends with `open(..., "a")` and a trace line runs to
# tens of kilobytes, well past the 4096-byte PIPE_BUF that makes an O_APPEND
# write atomic. Two collectors sharing `dataset/traces.jsonl` would interleave
# mid-line and corrupt records that look fine until `build_dataset.py` fails to
# parse them — or worse, parses a spliced one.
#
# Concurrency is capped by RAM, not by the provider: each process loads the full
# langchain/deepagents stack, and this machine has 8 GB with ~1.5 GB free. See
# evals/run_rebase.sh, where five concurrent eval processes were OOM-killed
# silently.
set -u
cd "$(dirname "$0")/../.."
PY=venv/bin/python3.12
LOGS="${1:?usage: collect_wave.sh <log-dir> <concurrency> <query-id>...}"; shift
CONC="${1:?missing concurrency}"; shift
mkdir -p "$LOGS"
DATA=finetune/pro_agent/dataset
STAGE="$LOGS/stage"

pids=(); n=0
for qid in "$@"; do
  out="$STAGE/$qid"
  mkdir -p "$out"
  echo "[$(date +%H:%M:%S)] start $qid -> $out"
  $PY -u finetune/pro_agent/collect.py --only "$qid" --out "$out" \
      > "$LOGS/$qid.log" 2>&1 &
  pids+=($!)
  n=$((n+1))
  if [ "$n" -ge "$CONC" ]; then
    for p in "${pids[@]}"; do wait "$p"; done
    pids=(); n=0
  fi
done
for p in "${pids[@]:-}"; do wait "$p" 2>/dev/null; done

# Merge only after every writer has exited, so the append is single-threaded.
for qid in "$@"; do
  for f in traces rejected; do
    src="$STAGE/$qid/$f.jsonl"
    [ -s "$src" ] && cat "$src" >> "$DATA/$f.jsonl"
  done
done
echo "[$(date +%H:%M:%S)] merged into $DATA/traces.jsonl"
for qid in "$@"; do
  k=$(wc -l < "$STAGE/$qid/traces.jsonl" 2>/dev/null || echo 0)
  r=$(wc -l < "$STAGE/$qid/rejected.jsonl" 2>/dev/null || echo 0)
  printf "  %-8s accepted=%s rejected=%s\n" "$qid" "$k" "$r"
done
