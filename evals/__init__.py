"""Omni behavioural evaluation harness.

Runs the real pro-mode agent against a fixed set of cases and scores what came
back on three independent layers:

- **A / deterministic** (`checks.py`) — did it load the right skill, call the
  right tools, emit a well-formed `<report>` / ```echarts / ```map /
  `<question>` block, obey the prompt's format rules, cite honestly.
- **B / judge** (`judge.py`) — an independent LLM scoring the things a regex
  cannot: did it answer, is the substance real, did it follow the skill's
  workflow, are the numbers the ones the tools actually returned.
- **C / metrics** (`trace.py`) — TTFT, latency, turns, tokens, cost. Recorded,
  never scored.

Every query, threshold, weight and judge prompt lives in `cases.yaml`; the code
here only implements *how* a check is evaluated, never *what* it expects.

See PLAN.md for the design rationale.
"""
