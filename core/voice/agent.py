"""The voice agent: langchain's `create_agent` (a LangGraph ReAct agent), not
deepagents and not a hand-rolled `bind_tools` loop, checkpointed the same way
as every other agent in this app.

`core/agent.py`'s `create_deep_agent` harness carries skills, report writing,
a 30-call budget — all dead weight for a spoken turn where every extra round
trip is latency the user is sitting through in silence. `create_agent` gets a
plain ReAct loop with exactly the two tool families the voice prompt
(`core/voice/prompt.py`) knows about, still wired into the same Redis-backed
checkpointer (`core/database/checkpointer.py`) and the same
config["configurable"]["thread_id"] convention core/agent.py's agents use —
conversation state lives in that checkpoint, not in a `history` list threaded
by hand through `core/voice/session.py`.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

import core.database.checkpointer as _db
from core.llm import gpt_oss_120b_low
from core.stream import _tagged_block
from core.tools.adapters import web_search, weather_current, weather_forecast
from core.utils.data_model import Personalization
from core.utils.utils import format_system_reminder
from core.voice.prompt import VOICE_SYSTEM_PROMPT

# Bare callables — create_agent converts these into StructuredTools itself
# (schema from the type hints + docstring), the same functions core/agent.py
# hands its own agents.
VOICE_TOOLS = [web_search, weather_current, weather_forecast]

# Built once at startup by initialize_voice_agent(), called from main.py's
# lifespan right after initialize_agents() — it needs _db.checkpointer, which
# only exists once setup_checkpointer() has run.
voice_agent = None


def initialize_voice_agent() -> None:
    global voice_agent
    voice_agent = create_agent(
        model=gpt_oss_120b_low,
        tools=VOICE_TOOLS,
        system_prompt=VOICE_SYSTEM_PROMPT,
        checkpointer=_db.checkpointer,
        middleware=[ToolCallLimitMiddleware(run_limit=2)],
    )


async def _repair_dangling_tool_calls(config: dict) -> None:
    """Patch a checkpoint left mid-tool-call by a barge-in.

    Barge-in cancels a turn's asyncio.Task outright (see
    core/voice/session.py's `_barge_in`), and that cancellation can land
    while the "tools" node is still awaiting a real tool call (a web search,
    a weather fetch). LangGraph checkpoints the *model* node's AIMessage
    (tool_calls included) as soon as that superstep finishes — independent
    of whether the following tool node ever completes — so a turn cancelled
    at exactly that point leaves the checkpoint with an AIMessage whose
    tool_calls were never answered. LangGraph's own ToolNode always appends
    every pending call's ToolMessage together in one state update, so this
    is genuinely all-or-nothing: if the *last* message is a tool-calling
    AIMessage, none of its calls have a response yet.

    Left alone, every later turn on this thread fails outright: replaying
    that history to the model 400s ("an assistant message with tool_calls
    must be followed by tool messages ..."), which — since core/voice/
    session.py now reuses one thread_id across a whole call, not a
    call-scoped history — would otherwise permanently wedge the thread
    after a single mistimed interruption. Confirmed live: an interrupted
    tool call reliably reproduces this 400 on the very next turn.
    """
    state = await voice_agent.aget_state(config)
    messages = state.values.get("messages") if state.values else None
    if not messages:
        return
    last = messages[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return
    patch = [
        ToolMessage(content="Cancelled — the user interrupted before this finished.", tool_call_id=call["id"])
        for call in last.tool_calls
    ]
    await voice_agent.aupdate_state(config, {"messages": patch})


def _build_turn_content(text: str, user_location: str | None, user_local_datetime: str | None) -> str:
    """Same personalization framing the main agent uses — `format_system_reminder`
    (core/utils/utils.py) plus `build_message_content`'s `<system_reminder>`/
    `<user_query>` tag wrapping (core/stream.py) — reused as-is rather than
    reimplemented, just fed a `Personalization` with only the two fields the
    voice client actually has (see hooks/useVoiceSession.ts): no
    `response_language`/`memory_enabled`, this is a live call, not a chat
    turn with its own settings panel and stored memory.
    """
    # response_language explicitly blanked out: Personalization defaults it to
    # a non-empty "Follow User's Query Language" (which format_system_reminder
    # would then include), but voice has no language-preference setting to
    # source that from, and the prompt already tells the model to mirror
    # whatever language the user just spoke.
    personalization = Personalization(
        response_language="", user_location=user_location, user_local_datetime=user_local_datetime
    )
    system_reminder = format_system_reminder(personalization)
    return "\n\n".join(
        block
        for block in (_tagged_block("system_reminder", system_reminder), _tagged_block("user_query", text))
        if block
    )


async def run_voice_turn(
    thread_id: str,
    text: str,
    *,
    user_location: str | None = None,
    user_local_datetime: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run one user turn to completion, streaming events as they happen.

    Yields:
        {"type": "text", "delta": str}
        {"type": "tool_start", "name": str, "args": dict}
        {"type": "tool_end", "name": str}
        {"type": "done"}  # always last

    Only the new user message goes in — the checkpointer already holds every
    prior turn under `thread_id`, so there's nothing else to pass in or hand
    back (contrast the old hand-threaded `history: list[BaseMessage]`).
    """
    config = {"configurable": {"thread_id": thread_id}}
    await _repair_dangling_tool_calls(config)
    content = _build_turn_content(text, user_location, user_local_datetime)
    input_state = {"messages": [HumanMessage(content)]}

    announced_tool_calls: set[str] = set()
    async for mode, data in voice_agent.astream(input_state, config=config, stream_mode=["messages", "updates"]):
        if mode == "messages":
            chunk = data[0] if isinstance(data, tuple) else data
            if isinstance(chunk, AIMessageChunk) and chunk.content:
                yield {"type": "text", "delta": chunk.content}
        elif mode == "updates" and isinstance(data, dict):
            for node_output in data.values():
                if not isinstance(node_output, dict):
                    continue
                messages = node_output.get("messages")
                if not messages:
                    continue
                if not isinstance(messages, list):
                    messages = [messages]
                for msg in messages:
                    for call in getattr(msg, "tool_calls", None) or []:
                        if call["id"] not in announced_tool_calls:
                            announced_tool_calls.add(call["id"])
                            yield {"type": "tool_start", "name": call["name"], "args": call["args"]}
                    if getattr(msg, "type", None) == "tool":
                        yield {"type": "tool_end", "name": msg.name}

    yield {"type": "done"}
