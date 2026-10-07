"""The interactive agent's tool budget: how many tool calls a turn may spend, and
what happens when it runs out or starts going in circles.

Replaces `ToolCallLimitMiddleware(run_limit=30)`, which allowed 30 calls to
everything and, once they were spent, answered each further call with a "limit
exceeded" tool result. A model in that state does not stop: the teacher retried
eight times at `run_limit=6` before writing its answer (see
`finetune/pro_agent/collect.py::trim_blocked_retries`). This one takes the tools
away instead, so the only thing left to do is answer.

## The rules

1. **Budget.** A turn gets `LIGHT_RUN_LIMIT` (10) tool calls. If a research skill
   (`HEAVY_SKILLS`: web-research, trip-advisor) has been loaded, it gets
   `HEAVY_RUN_LIMIT` (30). "Loaded" means anywhere in the conversation, not only
   this turn: the skill's instructions are still in the model's context, so a
   follow-up inside a research thread should not be squeezed back to 10. Loading
   a skill does not itself spend budget — it is fixed set-up cost, not research.
2. **No loops.** If the last three tool calls of the turn have the same tool and
   exactly the same arguments, the tools are taken away.
3. **Forced answer.** Taking the tools away means: the model call goes out with an
   empty tool list and one added line telling it so, and whatever tool calls it
   emits anyway are discarded. If it answers with nothing but tool calls it gets
   one retry with an explicit "reply in plain text" message, and only then a stock
   sentence. There is no limit on model calls themselves.

A "turn" is everything since the last user message, which is also what
`ToolCallLimitMiddleware` counted. The count is read off the messages on every
model call rather than stored, so a rewind or a resumed thread cannot leave it
stale.

## Two ways a skill gets loaded

The agent reads `/skills/<name>/SKILL.md` itself (a `read_file` tool call), *or*
the app hands the skill over without a tool call — the user picked it, or the
intent router matched the request — by putting its body in the user message with
"The <name> skill was already loaded for you" (core/context_enrichment.py,
`requested_skill_enrichment`). Both count; checking only tool calls would leave a
user who explicitly picked Deep Research on the 10-call budget.

## Known limits

- A single model message can emit several parallel tool calls, so a turn can
  overshoot its limit by that batch before the next model call is intercepted.
- The forced-answer note changes the system prompt for that one call. It is the
  only place this module touches the prompt, and it never appears on a normal call.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable

from deepagents.middleware._utils import append_to_system_message
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

LIGHT_RUN_LIMIT = 10
HEAVY_RUN_LIMIT = 30
HEAVY_SKILLS = frozenset({"web-research", "trip-advisor"})
REPEAT_LIMIT = 3

FORCED_ANSWER_NOTE = (
    "You can no longer call tools — none are available. Answer now from what you "
    "have already gathered, and say plainly where the evidence is incomplete."
)
# Sent as a last user message on the one retry, when a model handed no tools still
# answers with nothing but tool calls (seen: Gemini, about one forced call in
# three — it mimics the calls in its own history).
RETRY_NUDGE = (
    "Tools are not available. Reply now in plain text with your best answer from "
    "the information above."
)
# Used only if even the retry produces nothing to say.
FALLBACK_ANSWER = (
    "I ran out of room to keep searching before I could put a complete answer "
    "together. Ask me again, or narrow the question, and I'll pick it up from here."
)

# The metadata key stamped on a forced-answer message, so evals and logs can tell
# a budget stop from a model that simply chose to answer.
METADATA_KEY = "tool_budget"

_SKILL_PATH_RE = re.compile(r"/skills/([^/\s\"']+)/")
_PRELOADED_RE = re.compile(r"The ([A-Za-z0-9_-]+) skill was already loaded for you")
_REQUESTED_RE = re.compile(r"<requested_skill>\s*([A-Za-z0-9_-]+)\s*</requested_skill>")


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _args_key(call: dict) -> str:
    return json.dumps(call.get("args") or {}, sort_keys=True, ensure_ascii=False, default=str)


def _skill_of_call(call: dict) -> str | None:
    """The skill a `read_file` call opens, if it opens one."""
    if call.get("name") != "read_file":
        return None
    m = _SKILL_PATH_RE.search(_args_key(call))
    return m.group(1) if m else None


def skills_loaded(messages: list[BaseMessage]) -> set[str]:
    """Every skill loaded anywhere in `messages`, by either route (see the module
    docstring)."""
    found: set[str] = set()
    for m in messages:
        if isinstance(m, AIMessage):
            for call in m.tool_calls or []:
                if (skill := _skill_of_call(call)) is not None:
                    found.add(skill)
        elif isinstance(m, HumanMessage):
            text = _text_of(m.content)
            found.update(_PRELOADED_RE.findall(text))
            found.update(_REQUESTED_RE.findall(text))
    return found


def current_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Messages after the last user message."""
    last_human = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
    return messages[last_human + 1:]


def run_limit_for(messages: list[BaseMessage]) -> int:
    return HEAVY_RUN_LIMIT if skills_loaded(messages) & HEAVY_SKILLS else LIGHT_RUN_LIMIT


def forced_reason(messages: list[BaseMessage]) -> str | None:
    """Why the tools should be taken away on the next model call, or None."""
    calls = [c for m in current_turn(messages) if isinstance(m, AIMessage) for c in (m.tool_calls or [])]

    if len(calls) >= REPEAT_LIMIT:
        tail = calls[-REPEAT_LIMIT:]
        if len({(c.get("name"), _args_key(c)) for c in tail}) == 1:
            return "repeat"

    # Opening a skill is set-up, not research, so it is not counted.
    spent = sum(1 for c in calls if _skill_of_call(c) is None)
    if spent >= run_limit_for(messages):
        return "limit"
    return None


def _messages_of(result: Any) -> list[BaseMessage]:
    if isinstance(result, ExtendedModelResponse):
        return result.model_response.result
    if isinstance(result, ModelResponse):
        return result.result
    return [result]


def _has_text(result: Any) -> bool:
    return any(isinstance(m, AIMessage) and _text_of(m.content).strip() for m in _messages_of(result))


def _finish(result: Any, reason: str) -> Any:
    """Make a forced-answer response safe to hand back to the graph: no tool
    calls (the tools node would run them — it still knows every tool), something
    to say, and a stamp saying why."""
    for msg in _messages_of(result):
        if not isinstance(msg, AIMessage):
            continue
        if msg.tool_calls:
            logger.warning(f"[tool_budget] dropped {len(msg.tool_calls)} tool call(s) from a forced answer")
            msg.tool_calls = []
        msg.invalid_tool_calls = []
        if not _text_of(msg.content).strip():
            msg.content = FALLBACK_ANSWER
        msg.response_metadata = {**(msg.response_metadata or {}), METADATA_KEY: f"forced:{reason}"}
    return result


class ToolBudgetMiddleware(AgentMiddleware):
    """See the module docstring."""

    def _forced_request(self, request: ModelRequest) -> tuple[ModelRequest, str | None]:
        reason = forced_reason(request.messages)
        if reason is None:
            return request, None
        logger.info(f"[tool_budget] forcing an answer ({reason}); {len(request.tools)} tool(s) withheld")
        return (
            request.override(
                tools=[],
                tool_choice=None,
                system_message=append_to_system_message(request.system_message, FORCED_ANSWER_NOTE),
            ),
            reason,
        )

    @staticmethod
    def _retry_request(request: ModelRequest) -> ModelRequest:
        return request.override(messages=[*request.messages, HumanMessage(content=RETRY_NUDGE)])

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Any]) -> Any:
        request, reason = self._forced_request(request)
        result = handler(request)
        if not reason:
            return result
        if not _has_text(result):
            logger.warning("[tool_budget] forced call produced no text; retrying once")
            result = handler(self._retry_request(request))
        return _finish(result, reason)

    async def awrap_model_call(
        self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[Any]]
    ) -> Any:
        request, reason = self._forced_request(request)
        result = await handler(request)
        if not reason:
            return result
        if not _has_text(result):
            logger.warning("[tool_budget] forced call produced no text; retrying once")
            result = await handler(self._retry_request(request))
        return _finish(result, reason)
