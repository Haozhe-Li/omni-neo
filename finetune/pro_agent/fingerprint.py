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

from deepagents import create_deep_agent  # noqa: E402
from langchain.agents.middleware import AgentMiddleware, ToolCallLimitMiddleware  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from core.agent import (  # noqa: E402
    SYSTEM_PROMPT,
    SKILL_FILES,
    RETRIEVAL_TOOLS,
    SKILLS_SOURCE,
    _register_harness_profiles,
)

HERE = Path(__file__).resolve().parent
BLESSED = HERE / "fingerprint.json"


class _Captured(Exception):
    """Raised from the middleware to stop before any model call is made."""


class _Capture(AgentMiddleware):
    """Intercept the fully-assembled request and abort.

    Sits in `wrap_model_call` because that is the last point where the prompt
    and tool list are exactly what the provider will receive.
    """

    def __init__(self) -> None:
        super().__init__()
        self.system = ""
        self.tools: list[dict] = []

    def _grab(self, request) -> None:
        system = getattr(request, "system_prompt", None)
        if not system:
            messages = list(getattr(request, "messages", []) or [])
            if messages and getattr(messages[0], "type", "") == "system":
                system = messages[0].content
        self.system = system or ""
        specs = []
        for tool in getattr(request, "tools", None) or []:
            try:
                specs.append(convert_to_openai_tool(tool))
            except Exception:
                specs.append({"function": {"name": getattr(tool, "name", str(tool))}})
        self.tools = specs
        raise _Captured

    def wrap_model_call(self, request, handler):
        self._grab(request)

    async def awrap_model_call(self, request, handler):
        self._grab(request)


def capture() -> tuple[str, list[dict]]:
    """Assemble the pro agent and return `(system_prompt, tool_schemas)`.

    Uses the real `chat_llm`, not a stub: deepagents resolves its harness profile
    from the model's `ls_provider`, and an unregistered provider would restore
    `BASE_AGENT_PROMPT` — fingerprinting a prompt nothing is ever served. No
    request is made; the middleware aborts first.
    """
    import asyncio

    from core.llm import chat_llm

    _register_harness_profiles()
    cap = _Capture()
    agent = create_deep_agent(
        name="omni-eval-pro",
        model=chat_llm,
        tools=RETRIEVAL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        skills=[SKILLS_SOURCE],
        checkpointer=InMemorySaver(),
        middleware=[cap, ToolCallLimitMiddleware(run_limit=30)],
    )

    async def _drive():
        state = {"messages": [{"role": "user", "content": "x"}], "files": SKILL_FILES}
        async for _ in agent.astream(state, config={"configurable": {"thread_id": "fp"}}):
            pass

    try:
        asyncio.run(_drive())
    except Exception:
        pass  # _Captured, possibly wrapped by langgraph
    if not cap.system:
        raise SystemExit("fingerprint: captured nothing — middleware never ran")
    return cap.system, cap.tools


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def compute() -> dict:
    import deepagents

    system, tools = capture()
    # Sorted and separator-normalised so a reordering of the tool list, which
    # changes nothing the model sees semantically, does not read as drift.
    tools_canon = json.dumps(sorted(tools, key=lambda t: json.dumps(t, sort_keys=True)),
                             sort_keys=True, separators=(",", ":"), ensure_ascii=False)
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
