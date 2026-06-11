"""Single all-around Omni agent, in two profiles.

`build_agent("fast")` returns a lean LangChain agent (Groq gpt-oss-20b, few
turns, no skills). `build_agent("pro")` returns a deepagents deep agent (Gemini,
generous turns, planning + the `write_report` tool + on-demand skills).

Both share ONE base prompt. The difference is purely capability: pro additionally
gets the report tool and a set of skills (deep-research / report-writing /
charting) that are surfaced via progressive disclosure — only their name +
description sit in the prompt; the full instructions are read on demand. This
keeps pro from over-reaching (writing a report for every little thing) while
still being able to go deep when the task calls for it.
"""

from __future__ import annotations

import os
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ToolRetryMiddleware,
    TodoListMiddleware,
)
from langchain_groq import ChatGroq
from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data

from core.database.postgresql_saver import checkpointer
from core.tools.web_search import google_search, google_search_places
from core.tools.web_page_reader import load_web_page
from core.tools.weather_tool import get_weather
from core.tools.stock_data_retriever import get_stock_data
from core.tools.currency_tool import get_realtime_currency_rate
from core.tools.search_document import read_user_document
from core.tools.artifact_tools import write_report

Profile = Literal["fast", "pro"]

# Retrieval tools shared by both profiles.
RETRIEVAL_TOOLS = [
    google_search,
    load_web_page,
    google_search_places,
    get_weather,
    get_stock_data,
    get_realtime_currency_rate,
    read_user_document,
]

# Report tool — pro only (charting is done inline via ```echarts fences, taught
# by the charting skill, so it needs no tool).
REPORT_TOOLS = [write_report]


# ── Skills (deepagents progressive disclosure) ──────────────────────────────
# Loaded from disk at startup and handed to the pro agent's StateBackend at
# stream time via `files=` (see core/stream.py). Source dir is the virtual
# "/skills/" path; each skill lives at "/skills/<name>/SKILL.md".
SKILLS_SOURCE = "/skills/"
_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", "skills")


def _load_skill_files() -> dict:
    files: dict = {}
    base = os.path.abspath(_SKILLS_DIR)
    if not os.path.isdir(base):
        return files
    for name in sorted(os.listdir(base)):
        md_path = os.path.join(base, name, "SKILL.md")
        if os.path.isfile(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                files[f"/skills/{name}/SKILL.md"] = create_file_data(f.read())
    return files


SKILL_FILES = _load_skill_files()


_BASE_PROMPT = """
<identity>
You are Omni, a capable, friendly, and thorough AI assistant. You answer clearly
and completely, reason carefully, and prefer verified information over guesswork.
</identity>

<retrieval_policy>
Unless the request is pure chit-chat, assume up-to-date or factual grounding may
be needed. Prefer your tools over memory:
- Facts / current events / specifics → `google_search`, then `load_web_page` to
  read the most relevant results.
- Local places, venues, businesses → `google_search_places`.
- Weather → `get_weather`. Stocks → `get_stock_data`. FX rates →
  `get_realtime_currency_rate`.
- Questions about a user-uploaded file → `read_user_document`.
Cite what you used naturally in prose.
</retrieval_policy>

<quality_bar>
Give substantive, genuinely useful answers — never terse or perfunctory. Explain
the "why", include relevant detail, concrete examples, and light structure (short
paragraphs, lists where they help). Match depth to the question: a simple factual
ask gets a tight, complete answer; an open-ended or how-to question gets a fuller,
well-organized one (typically several developed paragraphs). Don't pad, but never
be lazy or one-liner-ish.
</quality_bar>

<planning>
The MOMENT you realize a request needs a tool — web search, reading pages,
weather/stocks/FX, a user file, or producing a report — you MUST call `write_todos`
BEFORE that first tool call, to lay out the plan (1–6 concrete steps; even a
single-step task gets a one-item list). Then work the plan: mark a todo in_progress
before its work and completed after. Skip todos ONLY for pure chit-chat or an
answer you can give directly with no tools at all.
</planning>

<formatting>
Reply in Markdown. Use `$...$` for inline math and `$$...$$` for display math —
no other LaTeX delimiters. Warm, direct, natural tone. Don't restate the question.

NEVER draw charts, plots, graphs, or diagrams as ASCII / text art inside a code
block — it looks bad. When a visual genuinely helps, use a real chart if you have
that ability; otherwise use a clean Markdown table or just describe it in prose.
</formatting>
"""

# Registered with the prompt-leakage guard.
SYSTEM_PROMPTS = [_BASE_PROMPT]


def build_agent(profile: Profile):
    """Construct an Omni agent for the given profile."""
    if profile == "fast":
        model = ChatGroq(
            model="openai/gpt-oss-20b",
            reasoning_effort="medium",
            api_key=os.getenv("GROQ_API_KEY"),
        )
        return create_agent(
            model=model,
            tools=RETRIEVAL_TOOLS,
            system_prompt=_BASE_PROMPT,
            name="Omni Fast",
            checkpointer=checkpointer,
            middleware=[
                # write_todos, so fast can also plan when a task needs tools.
                TodoListMiddleware(),
                ToolRetryMiddleware(max_retries=1),
                ModelCallLimitMiddleware(run_limit=8),
            ],
        )

    if profile == "pro":
        return create_deep_agent(
            name="Omni Pro",
            model="google_genai:gemini-flash-latest",
            tools=RETRIEVAL_TOOLS + REPORT_TOOLS,
            system_prompt=_BASE_PROMPT,
            skills=[SKILLS_SOURCE] if SKILL_FILES else None,
            checkpointer=checkpointer,
            middleware=[
                ToolRetryMiddleware(
                    max_retries=2,
                    backoff_factor=2.0,
                    initial_delay=1.0,
                ),
                # Generous so the deep-research workflow (plan → many searches →
                # report) can finish; normal pro answers use only a few.
                ModelCallLimitMiddleware(run_limit=40),
            ],
        )

    raise ValueError(f"Unknown agent profile: {profile!r}")


# Eagerly built singletons (agents are stateless across threads thanks to the
# checkpointer, so one instance per profile is enough).
fast_agent = build_agent("fast")
pro_agent = build_agent("pro")


def get_agent(profile: Profile):
    """Return the prebuilt agent for ``profile``."""
    if profile == "pro":
        return pro_agent
    return fast_agent
