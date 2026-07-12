"""Parse a casual natural-language request into a structured scheduled task.

Backs the "Create a scheduled research" quick box in Settings — the user
types something like "send me a daily summary of AI news every morning" and
this turns it into the same {title, instruction, schedule_time} shape the
manual create form fills in by hand, which the frontend then loads into that
form for the user to review/adjust before saving (see core/routers/
scheduled_tasks.py's /schedule_task/parse and lib/cron.ts on the frontend).
"""

from __future__ import annotations

import logging
from typing import Literal

from langsmith import tracing_context
from pydantic import BaseModel, Field

from core.llm import research_schedule_llm

logger = logging.getLogger(__name__)


class ScheduleTime(BaseModel):
    frequency: Literal["daily", "weekly", "monthly"] = Field(
        description="How often to run this research. Never more often than daily."
    )
    time: str = Field(
        description='Local time of day to run, 24-hour "HH:MM" (e.g. "08:00"). '
        "Infer a sensible default if the user didn't give one."
    )
    weekday: int | None = Field(
        default=None,
        description="0=Sunday..6=Saturday. Set only when frequency is 'weekly', else null.",
    )
    day_of_month: int | None = Field(
        default=None,
        description="1-28. Set only when frequency is 'monthly', else null.",
    )


class ParsedSchedule(BaseModel):
    title: str = Field(description="A short (3-6 word) name for this scheduled task.")
    instruction: str = Field(
        description="The user's request rewritten as a clear, self-contained research "
        "instruction for an AI agent to execute unattended each time it fires."
    )
    schedule_time: ScheduleTime


_SYSTEM_PROMPT = """
You turn a casual request for recurring research into a structured scheduled task.

Output exactly three fields:
- title: a short (3-6 word) name for the task, title case, no punctuation.
- instruction: rewrite the request as a clear, self-contained research instruction for an AI research agent to execute UNATTENDED each time it fires — no "you asked" / conversational framing, just the research task itself. Expand vague requests into something concretely researchable.
- schedule_time: how often and roughly what local time of day to run it, inferred from the request. Never invent a frequency more often than daily. If no frequency is mentioned, default to daily. If no time is mentioned, pick a sensible default (morning news -> "08:00", evening wrap-up -> "18:00", otherwise "09:00").

Examples:
Request: Send me a daily summary of AI news every morning
Output: {"title": "AI News Digest", "instruction": "Research and summarize the most significant AI news and developments from the past 24 hours, covering major model releases, funding, research breakthroughs, and industry moves.", "schedule_time": {"frequency": "daily", "time": "08:00", "weekday": null, "day_of_month": null}}

Request: give me a stock market recap every Friday evening
Output: {"title": "Weekly Market Recap", "instruction": "Research and summarize this week's stock market performance, major index movements, and notable company or macroeconomic events.", "schedule_time": {"frequency": "weekly", "time": "18:00", "weekday": 5, "day_of_month": null}}

Request: monthly digest of space exploration news
Output: {"title": "Space Exploration Digest", "instruction": "Research and summarize the past month's major space exploration news, including launches, missions, and industry developments.", "schedule_time": {"frequency": "monthly", "time": "09:00", "weekday": null, "day_of_month": 1}}

Return ONLY a JSON object with exactly these three fields, no other text.
"""

# gpt-oss on Groq auto-camelCases the registered class name for tool-calling
# and fails Groq's strict tool-name validation — json_mode sidesteps that
# entirely (same fix as core/utils/source_credibility.py, core/utils/source_rerank.py).
_parser_model = research_schedule_llm.with_structured_output(ParsedSchedule, method="json_mode")


def parse_schedule_prompt(text: str) -> ParsedSchedule:
    """Structured parse of a casual request. Raises on failure — this backs
    an interactive create-flow, so the caller should surface the error
    rather than silently falling back to something the user didn't ask for."""
    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", f"Request: {text}"),
    ]
    with tracing_context(project_name="parse-schedule-prompt"):
        return _parser_model.invoke(messages)
