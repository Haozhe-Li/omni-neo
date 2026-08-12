"""Build the agent under test.

Mirrors `core/stream.py::_stream_agent`'s construction so the eval measures the
real thing — same `SYSTEM_PROMPT`, same tools, same skill files, same turn budget
— with exactly three deliberate departures, each documented at its call site
below.
"""
from __future__ import annotations

from datetime import datetime

from deepagents import create_deep_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ToolRetryMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from core.agent import (
    SKILL_FILES,
    SYSTEM_PROMPT,
    RETRIEVAL_TOOLS,
    SKILLS_SOURCE,
    _register_harness_profiles,
)

RUN_LIMIT = 30


def build_eval_agent(
    llm: BaseChatModel,
    *,
    tools: list | None = None,
):
    """Construct an Omni agent bound to `llm`.

    There is one profile. The eval used to build a `fast` variant too (shorter
    prompt, 8-call budget, one retry); it went away with the profile split in
    core/agent.py and nothing ever ran it — `--profile` defaulted to pro on
    every recorded run.

    Three departures from production:

    1. **`InMemorySaver` instead of the Upstash checkpointer.** Eval threads are
       throwaway; writing hundreds of them into the production checkpoint store
       would both pollute it and make every run depend on Upstash being up.

    2. **No `ModelFallbackMiddleware`.** This one matters. In production a
       Cerebras hiccup silently promotes Gemini and the user still gets an
       answer — correct there, ruinous here. A whole run of `gemma-4-31b-high`
       could be served by `gemini-flash-lite` and the results would look
       completely normal, attributing one model's behaviour to another with no
       signal anywhere that it happened. An eval must fail loudly instead, so
       the chain is dropped and a provider outage surfaces as `status='error'`.

    3. **`model` is a parameter.** `core.agent.build_agent` hardcodes
       `chat_llm`; the whole point here is the model matrix.

    Everything else — prompt, tools, skills, retry policy, run limit — is
    production's.
    """
    _register_harness_profiles()
    return create_deep_agent(
        # Slug, not a display name. LangGraph stamps the agent name onto the
        # messages it emits, and OpenAI validates `messages[].name` against
        # `^[^\s<|\\/>]+$` — so "Omni Eval (pro)" makes every OpenAI request
        # fail with a 400 the moment a second message exists, which reads as
        # the model being broken rather than the name being illegal.
        name="omni-eval",
        model=llm,
        tools=tools if tools is not None else RETRIEVAL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        skills=[SKILLS_SOURCE],
        checkpointer=InMemorySaver(),
        middleware=[
            ToolRetryMiddleware(max_retries=2, backoff_factor=2.0, initial_delay=1.0),
            ToolCallLimitMiddleware(run_limit=RUN_LIMIT),
        ],
    )


def skill_files() -> dict:
    return SKILL_FILES


def build_personalization(cfg: dict[str, str]) -> str:
    """Render the `<personalization>` block.

    The datetime is pinned in `cases.yaml` rather than read from the clock:
    several cases ask about "tomorrow" or "the current state of X", and a
    floating today would make them un-reproducible across runs and
    un-comparable across models evaluated on different days.

    `language` is omitted entirely when the case leaves it unset — not defaulted
    to English. That omission is the point: with no stated response language the
    model has to infer one from the query itself, which is what the
    language-matching cases test. Emitting a default would answer the question
    for it and the check would always pass.
    """
    when = cfg.get("datetime") or datetime.now().isoformat(timespec="seconds")
    lines = []
    language = (cfg.get("language") or "").strip()
    if language:
        lines.append(f"Response language: {language}")
    lines.append(f"User location: {cfg.get('location', 'Unknown')}")
    lines.append(f"User local date and time: {when}")
    return "\n".join(lines)


def build_user_message(query: str, personalization: str, *, skill: str | None = None) -> str:
    """Assemble the tagged user turn.

    Same block order as `core/stream.py::build_message_content` — memory,
    personalization, requested skill, then the query last. Rebuilt here rather
    than imported because the production function also resolves uploaded-file
    records out of the database, and no eval case has attachments; importing it
    would make every case depend on a DB round trip that returns nothing.
    """
    blocks = []
    if personalization.strip():
        blocks.append(f"<personalization>\n{personalization.strip()}\n</personalization>")
    if skill:
        blocks.append(f"<requested_skill>\n{skill}\n</requested_skill>")
    blocks.append(f"<user_query>\n{query.strip()}\n</user_query>")
    return "\n\n".join(blocks)
