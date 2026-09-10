"""Follow-up questions offered under a finished answer.

One cheap model call per turn, run *after* the answer has already streamed
(see `core/routers/chat.py`), so nothing the reader is waiting on blocks on it.

Only the turn that just finished is passed in. Earlier turns are deliberately
left out: what the reader wants next follows from the answer in front of them,
and a whole thread of context makes the model drift toward summarising the
conversation instead of extending it — while costing tokens on every turn.

Failure is not an error. The caller swallows it and the frontend falls back to
its own pool, so a slow or malformed response costs the reader nothing.
"""
from __future__ import annotations

import logging

from langsmith import tracing_context
from pydantic import BaseModel, Field

from core.llm import gpt_oss_120b_low_groq

logger = logging.getLogger(__name__)

# Context caps. The question is short by nature; the answer is truncated from
# the head because that is where its subject is stated — a tail slice tends to
# catch a closing pleasantry and produces suggestions about nothing.
_MAX_QUERY_CHARS = 500
_MAX_ANSWER_CHARS = 2000

MIN_QUESTIONS = 3
MAX_QUESTIONS = 4


class FollowUps(BaseModel):
    questions: list[str] = Field(
        description=(
            "3 to 4 short follow-up questions the user could ask next, "
            "written in the same language as the user's own question."
        )
    )


_SYSTEM_PROMPT = """\
You generate follow-up questions for a chat assistant.

You are given the user's question and the assistant's answer. Produce 3-4 \
questions the user would plausibly want to ask NEXT.

Rules:
- Write every question in the SAME language as the USER'S question. Never \
translate. If the user wrote Chinese, every question is in Chinese; if the \
user wrote English, every question is in English.
- Keep them SHORT — at most about 10 words, or about 15 characters for CJK. \
They are read as a list, not as prose.
- Each question must go somewhere NEW: deeper into a point, a concrete case, a \
comparison, a limitation, a next step. Never re-ask something the answer \
already covers.
- Phrase them as the user speaking to the assistant, not the assistant \
speaking to the user.
- No numbering, no quotes, no trailing explanations.

Examples:

User: How do I debounce a resize handler in React?
Questions: ["What about throttling instead?", "How do I clean it up on unmount?", \
"Does this work with ResizeObserver?"]

User: 有什么美国推荐的 espresso 咖啡豆
Questions: ["哪些豆子适合做奶咖？", "深烘和中烘怎么选？", "网上哪里买最划算？", "开封后怎么保存？"]
"""


_llm = gpt_oss_120b_low_groq.with_structured_output(FollowUps, method="json_schema")


def _clean(questions: list[str]) -> list[str]:
    """Trim, drop empties and duplicates, cap the count.

    The cap is here rather than left to the prompt because an over-long list is
    the one failure the UI cannot absorb: the section is sized for four rows.
    """
    seen: set[str] = set()
    out: list[str] = []
    for q in questions:
        q = (q or "").strip().strip('"').strip()
        if not q:
            continue
        key = q.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= MAX_QUESTIONS:
            break
    # Fewer than three reads as a broken list rather than a short one — let the
    # frontend's own pool take the turn instead.
    return out if len(out) >= MIN_QUESTIONS else []


async def get_follow_ups(query: str, answer: str) -> list[str]:
    """3-4 next questions for this turn, or `[]` if anything goes wrong."""
    query = (query or "").strip()[:_MAX_QUERY_CHARS]
    answer = (answer or "").strip()[:_MAX_ANSWER_CHARS]
    if not query or not answer:
        return []

    messages = [
        ("system", _SYSTEM_PROMPT),
        (
            "human",
            f"<user_question>\n{query}\n</user_question>\n\n"
            f"<assistant_answer>\n{answer}\n</assistant_answer>",
        ),
    ]
    try:
        with tracing_context(project_name="follow_ups"):
            res = await _llm.ainvoke(messages)
        return _clean(getattr(res, "questions", []) or [])
    except Exception:
        # Never surfaced to the user — the frontend has its own fallback.
        logger.warning("follow-up generation failed", exc_info=True)
        return []
