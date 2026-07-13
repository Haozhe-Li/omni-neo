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
    AgentMiddleware,
    ToolRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.structured_output import ProviderStrategy
from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from pydantic import BaseModel, Field

import core.database.postgresql_saver as _db
from core.tools.web_search import google_search, google_search_places
from core.tools.web_page_reader import load_web_page
from core.tools.weather_tool import get_weather, get_weather_forecast
from core.tools.stock_data_retriever import get_stock_data
from core.tools.currency_tool import get_realtime_currency_rate
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
NEVER answer from your own knowledge alone. For anything beyond pure chit-chat,
you MUST call a grounding tool — at minimum one `google_search` — before you
answer, even if you're already confident you know it. Confidence is not the
same as current or correct; treat your own knowledge as unverified until a
tool backs it up. Route by topic:
- Facts / current events / specifics → `google_search`, then `load_web_page` to
  read the most relevant results.
- Local places, venues, businesses → `google_search_places`.
- Current weather only → `get_weather`. Forecasts / tomorrow / specific hours
  today / upcoming conditions → `get_weather_forecast` (returns current +
  today's 3-hour slots + tomorrow & day-after summaries). Stocks →
  `get_stock_data`. FX rates → `get_realtime_currency_rate`.
- Questions about a user-uploaded document → it's mounted at `/uploads/` in your
  filesystem; use `ls`, `read_file`, or `grep` to explore and read it.
- Exceptions (no search needed): pure computation/reasoning (see
  <computation_policy>) and creative writing — nothing external to verify.

Search discipline (hard limits — no exceptions):
- Per question or sub-topic: at most 2 `google_search` calls (one focused query +
  one reformulation if the first yields nothing useful). Never run a third search
  on the same sub-topic.
- Per search result: read at most 2 pages via `load_web_page`. Stop as soon as
  you have enough to answer — do not read for completeness.
- If results are still weak after 2 searches, answer with what you have and note
  the limitation. Do not keep searching.
</retrieval_policy>

<citation_policy>
Citing is MANDATORY whenever a claim, fact, figure, or quote in your answer
came from a `google_search`/`load_web_page` result (each carries a `n`) —
never skip it, no matter how obvious the fact seems. Facts you already knew,
or pure reasoning/opinion, need no citation.

Placement: never let citing interrupt the prose. Do NOT drop a [n] mid-sentence
or after every clause. Instead, batch all the [n]s a paragraph relies on into
one stack (e.g. [1][2]) at the very end of that paragraph, right before the
line break — only split a paragraph's citations into more than one cluster if
it makes two genuinely unrelated claims that a reader needs to tell apart.

Always ASCII `[`/`]`, never full-width (【】/［］), even in Chinese. Only use
`n` values that came from an actual `google_search`/`load_web_page` result —
either this turn's or an earlier turn's in this same conversation. You do NOT
need to re-run a search just to cite a source you already have a `[n]` for;
reuse that number. Never invent a number that wasn't given to you by a tool
result.
</citation_policy>

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
A turn is 100% tool call(s) or 100% final text — never both. If you're calling a
tool (including `write_todos`), output zero other content in that turn: no
preamble, no draft answer, no trailing remarks. Write your final answer only in a
later turn that contains no tool calls at all.
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
- The MOMENT a step's work is done, your VERY NEXT action MUST be a standalone
  `write_todos` call flipping it to `completed`, before starting the next step.
  Never leave a finished step unchecked, and never mark one completed early.
- Every todo must be `completed` before you write the final answer/report — and
  that last `write_todos` call is its own turn, per tool_call_discipline above.
</planning>

<formatting>
Reply in Markdown. Use `$...$` for inline math and `$$...$$` for display math —
no other LaTeX delimiters. Warm, direct, natural tone. Don't restate the question.

NEVER include hyperlinks of any form in your response unless the user explicitly
asks for a link or URL. Don't wrap text in `[text](url)` markdown links, don't
bare-print URLs. The one exception is the [n] citation markers described in
<citation_policy> — those are REQUIRED (not optional) whenever you cite a
source, and must be used instead of hyperlinks.

NEVER draw charts, plots, graphs, or diagrams as ASCII / UTF-8 text art inside a
code block — it always looks bad and must not appear. {chart_policy}
</formatting>
{artifact_policy}"""

# ── Scheduled profile ────────────────────────────────────────────────────────
# Deliberately NOT built from `_BASE_PROMPT` — an unattended cron run has a
# different output surface (three structured fields, no chat reply, no
# `<report>`/`<summary>` tags to stream to a reader pane) and doesn't need the
# interactive-only policies (artifact/chart-in-chat framing), so it gets its
# own prompt written for exactly what it does.
#
# Skills: everything the pro profile gets, minus three:
# - ask-question: no user present to answer a clarifying question in an
#   unattended cron run, so the agent must assume and proceed instead of
#   stalling the turn on it.
# - report-writing: teaches the `<report>…</report>` inline-streaming
#   convention, which doesn't apply here — the report is a schema field, not
#   something written inline and pulled out of the text after the fact (see
#   <output_contract> below).
# - deep-research: its plan/gather/reflect workflow is exactly what a
#   scheduled run needs, but being an optional, progressively-disclosed skill
#   made it easy for the agent to under-invest — a couple of shallow searches
#   and a thin report, never actually loading the skill. Baked directly into
#   <research_process> below instead of left optional, so every run gets the
#   full workflow (see build_scheduled_agent's docstring).
SCHEDULED_SKILL_FILES = {
    path: data for path, data in PRO_SKILL_FILES.items()
    if not path.startswith("/skills/ask-question/")
    and not path.startswith("/skills/report-writing/")
    and not path.startswith("/skills/deep-research/")
}


class ScheduledReportOutput(BaseModel):
    """Structured final output of a scheduled research run."""

    title: str = Field(description="Concise report title, max ~10 words.")
    summary: str = Field(
        description="Plain-text executive summary, 2-4 sentences, no markdown "
        "formatting and no [n] citations — this becomes the body of the "
        "notification email, so it must stand alone and make sense without "
        "the full report attached."
    )
    report: str = Field(
        description="The full report in GitHub-flavoured Markdown (##/### "
        "headings, lists, tables, $...$ / $$...$$ for math, optional "
        "```echarts fenced charts). Cite every claim drawn from a tool "
        "result with [n] per the citation policy. Do NOT wrap this in "
        "<report> or any other tag — this field IS the report body."
    )


_SCHEDULED_PROMPT = """
<identity>
You are Omni, running as an unattended scheduled research agent. A user set
this task up in advance to fire on a recurring schedule. Nobody is watching
this run live and there is no chat surface to reply in — your only output is
the structured report you produce at the end, delivered later by email.
</identity>

<retrieval_policy>
NEVER answer from your own knowledge alone. You MUST call a grounding tool —
at minimum one `google_search` — before writing the report, even if you're
already confident you know it. Confidence is not the same as current or
correct; treat your own knowledge as unverified until a tool backs it up.
Route by topic:
- Facts / current events / specifics → `google_search`, then `load_web_page`
  to read the most relevant results.
- Local places, venues, businesses → `google_search_places`.
- Current weather → `get_weather`. Forecasts → `get_weather_forecast`.
  Stocks → `get_stock_data`. FX rates → `get_realtime_currency_rate`.
</retrieval_policy>

<citation_policy>
Citing is MANDATORY whenever a claim, fact, figure, or quote in the report
came from a `google_search`/`load_web_page` result (each carries a `n`) —
never skip it, no matter how obvious the fact seems. Facts you already knew,
or pure reasoning, need no citation.

Placement: never let citing interrupt the prose. Batch all the [n]s a
paragraph relies on into one stack (e.g. [1][2]) at the very end of that
paragraph — never mid-sentence, never scattered after every clause.

Always ASCII `[`/`]`, never full-width (【】/［］). Only use `n` values that
came from an actual tool result this run. Never invent a citation number.
</citation_policy>

<computation_policy>
You MUST call `run_python` for arithmetic beyond trivial mental math,
statistics, comparisons, unit conversions, or any other numerical analysis —
never approximate in your head or make up numbers. `run_python` is
text-only; for visualisations use an ```echarts fence directly in the report.
</computation_policy>

<research_process>
This is a scheduled deep-research run, not a quick lookup — follow this full
arc every time, not a shortcut version of it:

1. Plan — call `write_todos` before any search. Structure as a research arc:
   - Orient: one broad search to map the landscape (key players, sub-topics,
     timeframe).
   - Dive: one todo per major sub-topic or angle (3-5 dives). Each narrow
     enough to answer in 2-3 searches.
   - Compare: if the task involves options or tradeoffs, add an explicit
     synthesis todo.
   - Report: always the final todo.
   Aim for 6-10 todos total — fewer is fine only for a genuinely narrow scope.

2. Gather — for each todo:
   - One targeted `google_search` per sub-topic; `load_web_page` only on
     clearly relevant, non-paywalled results.
   - Read 2-4 pages per sub-topic. Stop once two consecutive pages add
     nothing new.
   - Hard cap: 2 searches per todo (initial + one reformulation). Never a
     third — move on with what you have.
   - Prefer primary sources and established outlets over aggregator
     summaries. If sources disagree, surface the disagreement — don't
     silently pick one.
   - Todo hygiene (strict): mark a todo `in_progress` immediately before
     starting it. The moment its work is done, your very next action must be
     a standalone `write_todos` call marking it `completed`, before any other
     tool call. Per-todo tool cap: 5 tool calls max — if reached with no
     result, mark it completed and move on.

3. Reflect — after every 2-3 dives: is there a significant angle the sources
   haven't addressed? If yes, add a todo. If no, proceed. A quick gut-check,
   not a reason to keep searching.

Budget: 6-12 sources total across the whole run — a handful of strong sources
beats exhaustive searching. Hard stop: if approaching the tool-call limit,
skip remaining gather todos and write the report with what you already have —
a partial, honest report beats running out of steps mid-search.
</research_process>

<unattended_run_policy>
Nobody is present to answer a clarifying question. Never stall a turn
waiting on one; make the most reasonable assumption, state it plainly in the
report, and proceed.
</unattended_run_policy>

<output_contract>
Once every todo is completed and research is done, produce your final
output — exactly the three fields of your response schema. There is no other
output surface: no chat reply, no preamble, no `<report>`/`<summary>` tags
anywhere. The schema fields ARE the output.

Aim for a genuinely thorough `report` (typically 1000-1500 words) —
substantive and well-organized, covering every dive from your plan, never
terse or perfunctory, but no filler either. Embed at least 1-2 ```echarts
charts wherever data is clearer shown than told (comparisons, trends,
distributions). Never draw charts, plots, or diagrams as ASCII/UTF-8 text
art. Never include hyperlinks or bare URLs anywhere in the report — the [n]
citation markers are the only allowed reference to a source.
</output_contract>
"""


def build_scheduled_agent():
    """Construct the agent variant used for scheduled research tasks.

    Uses Gemini Flash-Lite with `ProviderStrategy` structured output (verified
    empirically to work alongside tool calls on this model) instead of asking
    the model to hand-wrap a `<summary>`/`<report>` block in free text — the
    prior tag-parsing approach was fragile (a malformed/missing tag silently
    broke `core/scheduled_agent.py`'s regex extraction) and is unrelated to
    why scheduled uses a cheaper model than pro: no one is watching this run
    live, so there's no latency pressure to justify a pricier one either way.
    """
    return create_deep_agent(
        name="Omni Scheduled",
        model=gemini_flash_lite_latest,
        tools=RETRIEVAL_TOOLS,
        system_prompt=_SCHEDULED_PROMPT,
        skills=[SKILLS_SOURCE] if SCHEDULED_SKILL_FILES else None,
        checkpointer=_db.checkpointer,
        response_format=ProviderStrategy(ScheduledReportOutput),
        middleware=[
            ToolRetryMiddleware(
                max_retries=2,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            ToolCallLimitMiddleware(run_limit=30),
        ],
    )


scheduled_agent = None


def get_scheduled_agent():
    return scheduled_agent


# Registered with the prompt-leakage guard.
SYSTEM_PROMPTS = [
    _BASE_PROMPT.format(chart_policy=_CHART_POLICY_FAST, artifact_policy=_ARTIFACT_POLICY_FAST),
    _BASE_PROMPT.format(chart_policy=_CHART_POLICY_PRO, artifact_policy=_ARTIFACT_POLICY_PRO),
    _SCHEDULED_PROMPT,
]


def _messages_have_image(messages) -> bool:
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, list) and any(
            isinstance(b, dict) and b.get("type") == "image_url" for b in content
        ):
            return True
    return False


class FastVisionModelMiddleware(AgentMiddleware):
    """Fast profile only: swap to a vision-capable model whenever an image
    appears anywhere in the conversation, not just the latest turn — the
    default fast model (gpt-oss-120b) is text-only and would error on
    multimodal content if a later turn re-references an earlier image."""

    def __init__(self, vision_model):
        super().__init__()
        self.vision_model = vision_model

    def wrap_model_call(self, request, handler):
        if _messages_have_image(request.messages):
            request = request.override(model=self.vision_model)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        if _messages_have_image(request.messages):
            request = request.override(model=self.vision_model)
        return await handler(request)


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
                FastVisionModelMiddleware(gemma_4_31b),
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
    global fast_agent, pro_agent, scheduled_agent
    fast_agent = build_agent("fast")
    pro_agent = build_agent("pro")
    scheduled_agent = build_scheduled_agent()


def get_agent(profile: Profile):
    if profile == "pro":
        return pro_agent
    return fast_agent
