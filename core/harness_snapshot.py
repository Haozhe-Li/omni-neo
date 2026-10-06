"""The assembled agent harness — the system prompt and tool schemas a model is
actually served — captured from the real agent rather than read off `SYSTEM_PROMPT`.

Two consumers, one implementation:

- `finetune/pro_agent/fingerprint.py` hashes it to detect drift against a blessed
  copy (the adapter's compatibility key).
- `core/sft_capture.py` stores it beside every thumbs-up example, so a training
  row can later be rebuilt under exactly the prompt it was generated with.

Why not just `SYSTEM_PROMPT`: deepagents appends ~1.5k tokens of its own sections
(`## write_todos`, `## Skills System`, `## Filesystem Tools`, ...) at request
time, and the skill roster is inlined into one of them. What the model reads is
only observable by letting the middleware stack assemble the request and
stopping before the model call, which is what `acapture` does. No request is made.
"""
from __future__ import annotations

import asyncio
import hashlib
import json

from deepagents import create_deep_agent
from langchain.agents.middleware import AgentMiddleware, ToolCallLimitMiddleware
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver

from core.agent import (
    AGENT_TOOLS,
    SKILL_FILES,
    SKILLS_SOURCE,
    SYSTEM_PROMPT,
    _register_harness_profiles,
)


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


async def acapture(model) -> tuple[str, list[dict]]:
    """Assemble the agent and return `(system_prompt, tool_schemas)`.

    `model` should be a real chat model, not a stub: deepagents resolves its
    harness profile from the model's `ls_provider`, and an unregistered provider
    would restore `BASE_AGENT_PROMPT` — capturing a prompt nothing is ever
    served. Every provider is registered, so any of our models yields the same
    result.
    """
    _register_harness_profiles()
    cap = _Capture()
    agent = create_deep_agent(
        name="omni-harness-capture",
        model=model,
        tools=AGENT_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        skills=[SKILLS_SOURCE],
        checkpointer=InMemorySaver(),
        middleware=[cap, ToolCallLimitMiddleware(run_limit=30)],
    )
    state = {"messages": [{"role": "user", "content": "x"}], "files": SKILL_FILES}
    try:
        async for _ in agent.astream(state, config={"configurable": {"thread_id": "harness-capture"}}):
            pass
    except Exception:
        pass  # _Captured, possibly wrapped by langgraph
    if not cap.system:
        raise RuntimeError("harness capture: middleware never ran")
    return cap.system, cap.tools


def capture(model) -> tuple[str, list[dict]]:
    """Sync wrapper for scripts. Must not be called from a running event loop."""
    return asyncio.run(acapture(model))


def canonical_tools(tools: list[dict]) -> str:
    """Tool schemas as one stable string. Sorted and separator-normalised so a
    reordering of the tool list, which changes nothing the model sees
    semantically, does not read as drift."""
    return json.dumps(
        sorted(tools, key=lambda t: json.dumps(t, sort_keys=True)),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def harness_hash(system: str, tools: list[dict]) -> str:
    """Short id for one (assembled prompt, tool schemas) pair."""
    h = hashlib.sha256()
    h.update(system.encode("utf-8"))
    h.update(b"\x00")
    h.update(canonical_tools(tools).encode("utf-8"))
    return h.hexdigest()[:16]


# ── cached live snapshot ────────────────────────────────────────────────────
# The harness is fixed for the life of a process (prompt, skills and tools are
# all read at import / startup), so it is assembled at most once.

_live: tuple[str, str, list[dict]] | None = None
_live_lock: asyncio.Lock | None = None


async def live_harness() -> tuple[str, str, list[dict]]:
    """`(harness_hash, system_prompt, tools)` for what this process serves."""
    global _live, _live_lock
    if _live is not None:
        return _live
    if _live_lock is None:
        _live_lock = asyncio.Lock()
    async with _live_lock:
        if _live is None:
            from core.llm import gpt_6_luna

            system, tools = await acapture(gpt_6_luna)
            _live = (harness_hash(system, tools), system, tools)
    return _live
