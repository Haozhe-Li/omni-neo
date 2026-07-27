"""LLM excerpt-extraction pass for `/check_source`.

Vector search already restricts candidates to chunks that score above a high
similarity threshold (see `vector_sources._MIN_SCORE`), so every candidate
handed here is presumed relevant — this module no longer decides keep/drop.
It makes one gpt-oss-20b call with every candidate chunk and asks it to copy
out the exact supporting phrase for each one verbatim. The frontend
fuzzy-matches that excerpt against the chunk text to render a pixel-precise
highlight, so the model must not translate, paraphrase, or alter it in any
way — only copy.

Every candidate is always returned. If the model can't find a clean
supporting excerpt for a candidate, is missing from the response, or returns
something that isn't a genuine substring of that candidate's text, the
candidate is still returned with `excerpt=""` rather than dropped — a source
match should degrade to "found the source, no snippet to highlight", never
to "no match at all".
"""
from __future__ import annotations

import logging

from langsmith import tracing_context
from pydantic import BaseModel, Field

from core.llm import gpt_oss_20b

logger = logging.getLogger(__name__)


class RerankItem(BaseModel):
    index: int = Field(description="The candidate's index, as given in the prompt.")
    excerpt: str = Field(
        default="",
        description="The exact contiguous substring copied character-for-character "
        "from this candidate's text that best supports the claim, in its "
        "original language — no translation or paraphrase. Empty string if "
        "no part of this candidate actually supports the claim.",
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

_SYSTEM_PROMPT = """You are extracting the exact sentence or phrase that supports a claim made by an AI assistant, from each of a list of candidate source passages.

You will be given a claim and a numbered list of candidate passages (already pre-filtered for high similarity to the claim). For EACH candidate, find the exact contiguous substring copied character-for-character from that candidate's text that best supports the claim. Copy it exactly as it appears, in its original language — do not translate, paraphrase, summarize, or alter it. Keep it as short as possible while still being the complete supporting phrase or sentence.

If a candidate genuinely contains no supporting text for the claim, set excerpt="" for it. Otherwise always produce an excerpt.

Return ONLY a JSON object of the form {"items": [{"index": int, "excerpt": str}]}, one item per candidate, indices matching the input."""


def _valid_excerpt(excerpt: str, chunk: str) -> bool:
    excerpt = excerpt.strip()
    if not excerpt:
        return False
    return excerpt in chunk


async def rerank_candidates(claim: str, candidates: list[dict]) -> list[dict]:
    """Attach a verbatim "excerpt" to every candidate in `candidates` (each a
    dict with a "chunk" text field), highlighting the substring that supports
    `claim`. Never drops a candidate: on any LLM failure, a missing item, or
    an excerpt that isn't a genuine substring of its chunk, that candidate is
    still returned with `excerpt=""`.
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

    excerpts_by_index: dict[int, str] = {}
    if result is not None:
        for item in result.items:
            if item.index < 0 or item.index >= len(candidates):
                continue
            excerpts_by_index[item.index] = item.excerpt

    matches = []
    for i, candidate in enumerate(candidates):
        excerpt = excerpts_by_index.get(i, "")
        if not _valid_excerpt(excerpt, candidate["chunk"]):
            excerpt = ""
        matches.append({**candidate, "excerpt": excerpt})
    return matches
