#!/bin/bash
# Re-measure every model in the picker on the current rubric, two processes at
# a time.
#
# Two at a time, not all at once: this machine has 8.6 GB and already runs a dev
# server plus the backend. Five concurrent eval processes — each loading the
# full langchain/deepagents stack — got OOM-killed ~8 cases in, silently, with
# no traceback and five half-written `status=running` rows left in Supabase.
#
# Grouped by provider so a wave never puts two models on the same vendor at
# once. Per-process case concurrency stays at the default 1: raising it is what
# produced 35 rate-limit failures previously, and the eval agent deliberately
# carries no fallback, so a 429 is recorded as the model failing rather than as
# an infrastructure problem.
set -u
cd "$(dirname "$0")/.."
PY=venv/bin/python3.12
LOGS="${1:?usage: run_rebase.sh <log-dir>}"
mkdir -p "$LOGS"

run_wave() {
  local pids=()
  for spec in "$@"; do
    local name="${spec%%:*}" models="${spec#*:}"
    echo "  start $name  ($models)"
    nohup $PY -m evals.cli --models $models \
      --label "rebase-2026-08-12" --out "bench_${name}.json" \
      > "$LOGS/$name.log" 2>&1 &
    pids+=($!)
  done
  local i=0
  for p in "${pids[@]}"; do
    wait "$p"; local rc=$?
    local name="${@:$((i+1)):1}"; name="${name%%:*}"
    if [ $rc -ne 0 ]; then echo "  !! $name exited $rc (137 = OOM-killed)"; else echo "  ok $name"; fi
    i=$((i+1))
  done
}

echo "wave 1/3"
run_wave "cerebras:gpt-oss-120b-low gemma-4-31b" "wandb:qwen3-30b-a3b rix-30b-a3b-v1 rix-30b-a3b-v3"
echo "wave 2/3"
run_wave "groq:gpt-oss-120b-low-groq gpt-oss-20b" "google:gemini-3-6-flash gemini-flash-lite-3-5"
echo "wave 3/3"
run_wave "openai:gpt-5-6-luna"
echo "ALL WAVES DONE"
