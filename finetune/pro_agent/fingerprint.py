"""Fingerprint the assembled agent harness — the adapter's compatibility key.

A LoRA only ever sees one system prompt and one tool schema. Whatever the agent
was assembled with at data-collection time is baked into the weights, so if the
assembled prompt or the tool list drifts afterwards, the adapter is being served
inputs it was never trained on and its scores stop meaning anything. The failure
is silent: nothing errors, the model just gets quietly worse.

What is fingerprinted, and why it is the *assembled* prompt rather than
`SYSTEM_PROMPT`:

- deepagents appends `## write_todos`, `## Skills System`, `## Filesystem Tools`
  and `## Large Tool Results` at request time — about 1,525 of the 4,368 tokens.
  Hashing `SYSTEM_PROMPT` alone would miss a deepagents upgrade entirely.
- The `## Skills System` section lists every skill's **name and description**
  inline. So the roster is covered for free, and the rule falls out of that:
  adding, removing or renaming a skill — or editing its `description:` — breaks
  the fingerprint, while editing a SKILL.md **body** does not. Bodies arrive at
  runtime as `read_file` results, which is training data, not weights.
- Tool schemas are rendered into the chat template, so a changed docstring is a
  changed prompt. All 15 are hashed, retrieval and deepagents-provided alike.

Usage:

    python finetune/pro_agent/fingerprint.py            # verify, non-zero on drift
    python finetune/pro_agent/fingerprint.py --update   # re-bless after an
                                                        # intentional change
    python finetune/pro_agent/fingerprint.py --show     # dump what is hashed

Re-blessing is not a formality. It means every trajectory collected under the
old fingerprint is stale, and any adapter trained on them has to be retrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
dotenv.load_dotenv(ROOT / ".env")

from core.agent import SYSTEM_PROMPT, SKILL_FILES  # noqa: E402
from core.harness_snapshot import acapture, canonical_tools  # noqa: E402

HERE = Path(__file__).resolve().parent
BLESSED = HERE / "fingerprint.json"


def capture() -> tuple[str, list[dict]]:
    """Assemble the pro agent and return `(system_prompt, tool_schemas)`.

    The assembly itself lives in `core/harness_snapshot.py`, shared with the
    thumbs-up capture so both read the harness through one implementation.
    Uses the real `chat_llm`, not a stub — see `acapture` for why.
    """
    import asyncio

    from core.llm import chat_llm

    try:
        return asyncio.run(acapture(chat_llm))
    except RuntimeError as e:
        raise SystemExit(f"fingerprint: {e}")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compute() -> dict:
    import deepagents

    system, tools = capture()
    tools_canon = canonical_tools(tools)
    return {
        "deepagents_version": getattr(deepagents, "__version__", "unknown"),
        "system_prompt_sha": _sha(system),
        "system_prompt_chars": len(system),
        "our_prompt_sha": _sha(SYSTEM_PROMPT),
        "tools_sha": _sha(tools_canon),
        "tool_names": sorted(
            t.get("function", {}).get("name", t.get("name", "?")) for t in tools
        ),
        "skill_files": sorted(SKILL_FILES),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true", help="re-bless the current harness")
    ap.add_argument("--show", action="store_true", help="print the assembled prompt and tools")
    args = ap.parse_args()

    current = compute()

    if args.show:
        system, tools = capture()
        print(system)
        print("\n\n===== TOOLS =====")
        print(json.dumps(tools, indent=2, ensure_ascii=False))
        return 0

    if args.update or not BLESSED.exists():
        BLESSED.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
        print(f"blessed -> {BLESSED.relative_to(ROOT)}")
        for k, v in current.items():
            if not isinstance(v, list):
                print(f"  {k}: {v}")
        if not args.update:
            print("\n! no prior fingerprint existed; nothing was verified")
        return 0

    blessed = json.loads(BLESSED.read_text())
    drift = [k for k in current if current[k] != blessed.get(k)]
    if not drift:
        print(f"harness unchanged  (prompt {current['system_prompt_sha']}, "
              f"tools {current['tools_sha']}, deepagents {current['deepagents_version']})")
        return 0

    print("HARNESS DRIFT — any collected trajectory or trained adapter is stale\n")
    for k in drift:
        b, c = blessed.get(k), current[k]
        if isinstance(c, list):
            added = sorted(set(c) - set(b or []))
            removed = sorted(set(b or []) - set(c))
            print(f"  {k}:")
            if added:
                print(f"    + {added}")
            if removed:
                print(f"    - {removed}")
        else:
            print(f"  {k}: {b} -> {c}")
    print("\nIf the change was intended, re-collect the data, then --update.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
