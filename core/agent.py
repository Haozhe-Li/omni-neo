"""Single all-around Omni agent, in two profiles.

Both profiles use deepagents and share ONE base prompt.  The difference is in
capability (model, turn budget) and the skill subset each receives:

- `build_agent("fast")` — gpt-oss-120b, tight turn budget, identity/info skills
  only (`about-omni`, `about-haozheli`).
- `build_agent("pro")` — Gemini Flash, generous turn budget, all skills
  (deep-research, report-writing, charting, …).

Skills are surfaced via progressive disclosure — only their name + description
sit in the prompt; full instructions are read on demand.  Charts and reports
stream inline (```echarts fences / `<report>…</report>` blocks).
"""

from __future__ import annotations

import os
from typing import Literal

from langchain.agents.middleware import (
    ToolRetryMiddleware,
    ToolCallLimitMiddleware,
)
from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data

import core.database.postgresql_saver as _db
from core.tools.web_search import google_search, google_search_places
from core.tools.web_page_reader import load_web_page
from core.tools.weather_tool import get_weather, get_weather_forecast
from core.tools.stock_data_retriever import get_stock_data
from core.tools.currency_tool import get_realtime_currency_rate
from core.tools.search_document import read_user_document
from core.tools.coding_sandbox import run_python
from core.llm import *

Profile = Literal["fast", "pro"]

# Retrieval tools shared by both profiles.
RETRIEVAL_TOOLS = [
    google_search,
    load_web_page,
    google_search_places,
    get_weather,
    get_weather_forecast,
    get_stock_data,
    get_realtime_currency_rate,
    read_user_document,
    run_python,
]

# Charts AND reports are produced inline in the answer stream (```echarts fences
# and `<report>…</report>` blocks), taught by the charting / report-writing
# skills — so neither needs a tool.


# ── Skills (deepagents progressive disclosure) ──────────────────────────────
# Loaded from disk at startup and handed to each agent's StateBackend at
# stream time via `files=` (see core/stream.py). Source dir is the virtual
# "/skills/" path; each skill lives at "/skills/<name>/SKILL.md".
#
# Fast profile gets a small allow-list of lightweight identity/info skills.
# Pro profile gets every skill.
SKILLS_SOURCE = "/skills/"
_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "..", "skills")

# Skills available in the fast profile (subset).
_FAST_SKILLS = {"about-omni", "about-haozheli"}


def _load_skill_files(only: set[str] | None = None) -> dict:
    """Load SKILL.md files from disk.

    Args:
        only: if given, only load skills whose directory name is in this set.
    """
    files: dict = {}
    base = os.path.abspath(_SKILLS_DIR)
    if not os.path.isdir(base):
        return files
    for name in sorted(os.listdir(base)):
        if only is not None and name not in only:
            continue
        md_path = os.path.join(base, name, "SKILL.md")
        if os.path.isfile(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                files[f"/skills/{name}/SKILL.md"] = create_file_data(f.read())
    return files


# Pre-loaded at startup; keyed by virtual path for the deepagents `files=` API.
FAST_SKILL_FILES = _load_skill_files(only=_FAST_SKILLS)
PRO_SKILL_FILES = _load_skill_files()  # all skills


_CHART_POLICY_FAST = (
    "In this (fast) profile you have no charting ability — do NOT produce any "
    "diagram or chart at all. Use a Markdown table or describe the data in prose."
)
_CHART_POLICY_PRO = (
    "In this (pro) profile, default to a chart over prose whenever the answer "
    "involves numbers, trends, comparisons, or distributions — use the charting "
    "skill. Never fall back to text art or a plain table when a chart is clearer."
)

_ARTIFACT_POLICY_FAST = ""

_ARTIFACT_POLICY_PRO = "" # skip for now

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
- Current weather only → `get_weather`. Forecasts / tomorrow / specific hours
  today / upcoming conditions → `get_weather_forecast` (returns current +
  today's 3-hour slots + tomorrow & day-after summaries). Stocks →
  `get_stock_data`. FX rates → `get_realtime_currency_rate`.
- Questions about a user-uploaded file → `read_user_document`.
Cite what you used naturally in prose.

Search discipline (hard limits — no exceptions):
- Per question or sub-topic: at most 2 `google_search` calls (one focused query +
  one reformulation if the first yields nothing useful). Never run a third search
  on the same sub-topic.
- Per search result: read at most 2 pages via `load_web_page`. Stop as soon as
  you have enough to answer — do not read for completeness.
- If results are still weak after 2 searches, answer with what you have and note
  the limitation. Do not keep searching.
</retrieval_policy>

<computation_policy>
You MUST call `run_python` for ANY of the following — never approximate in your
head or make up numbers:
- Arithmetic beyond trivial mental math (multi-step, fractions, large numbers).
- Statistics, probability, or data analysis of any kind.
- Unit conversions that require formula application.
- Numerical algorithms (sorting, searching, optimisation, simulation).
- Anything the user explicitly asks you to "calculate", "compute", "run",
  "simulate", or "verify with code".

`run_python` is text-only — it cannot produce charts or images. For visualisations
use the charting skill. Write one complete, self-contained script per call.
Do NOT use `run_python` for tasks that need no computation (explaining concepts,
translating text, etc.).
</computation_policy>

<quality_bar>
Give substantive, genuinely useful answers — never terse or perfunctory. Explain
the "why", include relevant detail, concrete examples, and light structure (short
paragraphs, lists where they help). Match depth to the question: a simple factual
ask gets a tight, complete answer; an open-ended or how-to question gets a fuller,
well-organized one (typically several developed paragraphs). Don't pad, but never
be lazy or one-liner-ish.
</quality_bar>

<tool_call_discipline>
If you are going to call any tool, do NOT output any text before the first tool
call. Make the tool call immediately and silently. Only produce text in your
final response after all tool work is done.
</tool_call_discipline>

<planning>
The MOMENT you realize a request needs a tool — web search, reading pages,
weather/stocks/FX, a user file, or producing a report — you MUST call `write_todos`
BEFORE that first tool call, to lay out the plan (1–6 concrete steps; even a
single-step task gets a one-item list). Skip todos ONLY for pure chit-chat or an
answer you can give directly with no tools at all.

Once you have a todo list, keep it honest and current — this is strict:
- Exactly ONE todo is `in_progress` at any time. Mark it `in_progress` before you
  start its work.
- The MOMENT a step's work is done, your VERY NEXT action MUST be a `write_todos`
  call that flips that todo to `completed` — before any other tool call, before
  starting the next step, and before writing your answer or report. Never carry on
  with a finished step still left unchecked, and never mark a step completed before
  its work is actually finished.
- By the time you write the final answer/report, every todo must be `completed`.
</planning>

<formatting>
Reply in Markdown. Use `$...$` for inline math and `$$...$$` for display math —
no other LaTeX delimiters. Warm, direct, natural tone. Don't restate the question.

NEVER include hyperlinks of any form in your response unless the user explicitly
asks for a link or URL. Don't wrap text in `[text](url)` markdown links, don't
bare-print URLs, don't cite sources as hyperlinks — reference sources by name in
prose instead.

NEVER draw charts, plots, graphs, or diagrams as ASCII / UTF-8 text art inside a
code block — it always looks bad and must not appear. {chart_policy}
</formatting>
{artifact_policy}"""

# Registered with the prompt-leakage guard.
SYSTEM_PROMPTS = [
    _BASE_PROMPT.format(chart_policy=_CHART_POLICY_FAST, artifact_policy=_ARTIFACT_POLICY_FAST),
    _BASE_PROMPT.format(chart_policy=_CHART_POLICY_PRO, artifact_policy=_ARTIFACT_POLICY_PRO),
]


def build_agent(profile: Profile):
    """Construct an Omni agent for the given profile."""
    if profile == "fast":
        return create_deep_agent(
            name="Omni Fast",
            model=fast_llm,
            tools=RETRIEVAL_TOOLS,
            system_prompt=_BASE_PROMPT.format(chart_policy=_CHART_POLICY_FAST, artifact_policy=_ARTIFACT_POLICY_FAST),
            skills=[SKILLS_SOURCE] if FAST_SKILL_FILES else None,
            checkpointer=_db.checkpointer,
            middleware=[
                ToolRetryMiddleware(max_retries=1),
                ToolCallLimitMiddleware(run_limit=8),
            ],
        )

    if profile == "pro":
        return create_deep_agent(
            name="Omni Pro",
            model=pro_llm,
            tools=RETRIEVAL_TOOLS,
            system_prompt=_BASE_PROMPT.format(chart_policy=_CHART_POLICY_PRO, artifact_policy=_ARTIFACT_POLICY_PRO),
            skills=[SKILLS_SOURCE] if PRO_SKILL_FILES else None,
            checkpointer=_db.checkpointer,
            middleware=[
                ToolRetryMiddleware(
                    max_retries=2,
                    backoff_factor=2.0,
                    initial_delay=1.0,
                ),
                ToolCallLimitMiddleware(run_limit=30),
            ],
        )

    raise ValueError(f"Unknown agent profile: {profile!r}")


fast_agent = None
pro_agent = None


def initialize_agents():
    global fast_agent, pro_agent
    fast_agent = build_agent("fast")
    pro_agent = build_agent("pro")


def get_agent(profile: Profile):
    if profile == "pro":
        return pro_agent
    return fast_agent
