"""LLM rerank pass for `/check_source`.

Vector search only proves a chunk is topically similar to the highlighted
claim — it doesn't prove the chunk actually supports it. This module makes
one gpt-oss-20b call with every candidate chunk and asks it to keep/drop each
one, and for keeps, copy out the exact supporting phrase verbatim. The
frontend fuzzy-matches that excerpt against the chunk text to render a
pixel-precise highlight, so the model must not translate, paraphrase, or
alter it in any way — only copy.
"""
from __future__ import annotations

import logging

from langsmith import tracing_context
from pydantic import BaseModel, Field

from core.llm import gpt_oss_20b

logger = logging.getLogger(__name__)


class RerankItem(BaseModel):
    index: int = Field(description="The candidate's index, as given in the prompt.")
    keep: bool = Field(
        description="True only if this passage genuinely supports the claim, "
        "not just topically related to it."
    )
    excerpt: str = Field(
        default="",
        description="If keep=true: the exact contiguous substring copied "
        "character-for-character from this candidate's text that supports "
        "the claim, in its original language — no translation or paraphrase. "
        "Empty string if keep=false.",
    )


class RerankResult(BaseModel):
    items: list[RerankItem]



# `method="json_mode"` instead of the default tool-calling structured output:
# gpt-oss-20b's tool-calling on Groq deterministically auto-camelCases the
# registered tool/function name in its generation (e.g. our "RerankResult"
# class comes back out as a call to "rerankResult"), which Groq's strict
# tool-name validation then rejects outright — 100% failure rate in testing.
# json_mode sidesteps tool-name matching entirely and was reliable across
# repeated trials.
_rerank_model = gpt_oss_20b.with_structured_output(RerankResult, method="json_mode")

_SYSTEM_PROMPT = """You are verifying which source passages actually support a claim made by an AI assistant.

You will be given a claim and a numbered list of candidate passages (already pre-filtered for topical similarity). For EACH candidate, decide:
- keep=true only if the passage genuinely supports or directly substantiates the claim. Being on the same topic is not enough.
- if keep=true, excerpt = the exact contiguous substring copied character-for-character from that candidate's text that best supports the claim. Copy it exactly as it appears, in its original language — do not translate, paraphrase, summarize, or alter it. Keep it as short as possible while still being the complete supporting phrase or sentence.
- if keep=false, excerpt = "".

Return ONLY a JSON object of the form {"items": [{"index": int, "keep": bool, "excerpt": str}]}, one item per candidate, indices matching the input."""


async def rerank_candidates(claim: str, candidates: list[dict]) -> list[dict]:
    """Filter `candidates` (each a dict with a "chunk" text field) down to the
    ones that genuinely support `claim`, attaching a verbatim "excerpt" to
    each survivor. Returns [] on any LLM failure — a rerank outage should
    degrade to no matches, not to unverified ones.
    """
    if not candidates:
        return []

    listing = "\n\n".join(
        f"[{i}] {c['chunk']}" for i, c in enumerate(candidates)
    )
    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", f"Claim:\n{claim}\n\nCandidates:\n{listing}"),
    ]

    # One retry as cheap insurance against a transient API/parse hiccup.
    result = None
    for attempt in range(2):
        try:
            with tracing_context(project_name="check_source_rerank"):
                result = await _rerank_model.ainvoke(messages)
            break
        except Exception as exc:
            logger.warning(f"[source_rerank] rerank attempt {attempt} failed: {exc}")
    if result is None:
        return []

    kept = []
    for item in result.items:
        if not item.keep:
            continue
        if item.index < 0 or item.index >= len(candidates):
            continue
        merged = {**candidates[item.index], "excerpt": item.excerpt}
        kept.append(merged)
    return kept
