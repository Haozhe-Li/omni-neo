"""Credibility classification for web sources (`google_search` / `load_web_page`).

Every source gets a `credibility` dict — `{"label": ..., "reason": ...}` — not
just a bare label. `reason` is always a one-sentence, human-readable
explanation; where it comes from depends on the layer that resolved the
source, cheapest first:

1. Regex on the hostname — unambiguous government/military/educational
   domains resolve to "official" for free, no cache, no LLM, reason is a
   templated sentence.
2. A Redis-backed domain whitelist (`redis_credibility.py`), bootstrapped by
   a small hardcoded seed list (wikipedia/google/apple/...) -> "trusted"
   with a templated reason. Lookups are batched with one MGET per call.
3. Whatever's left after (1) and (2) — including user-generated-content
   platforms (reddit, x.com, medium, ...), which are deliberately never
   resolved by (1)/(2) since quality varies wildly page-to-page — goes
   through a single batched gpt-oss-20b call (title + url + a short snippet
   + the user's query/topic, for judging "first_party"). The model produces
   both the label and its own one-sentence reason. Only "trusted" verdicts
   get written back into the Layer 2 cache (as JSON, `{label, reason}`
   together) — "official" is left to the regex layer, "first_party" is a
   property of (domain, query) not of the domain alone, "social_media" and
   "unknown" carry no reusable signal, and "junk" is deliberately never
   cached either: a page being junk for *this* query doesn't mean the whole
   domain is junk (could just be one marketing page on an otherwise fine
   site) — so every hit gets a fresh LLM judgment instead of inheriting a
   stale domain-wide verdict.
"""
from __future__ import annotations

import json
import logging
from typing import Literal, TypedDict
from urllib.parse import urlsplit

from langsmith import tracing_context
from pydantic import BaseModel, Field

from core.llm import credibility_llm
from core.utils.redis_credibility import TTL_TRUSTED, credibility_redis

logger = logging.getLogger(__name__)

CredibilityLabel = Literal[
    "official", "trusted", "first_party", "social_media", "junk", "unknown"
]


class Credibility(TypedDict):
    label: CredibilityLabel
    reason: str


# Suffixes that make a domain unambiguously official. Deterministic and
# free to check, so never cached. Extend as needed.
_OFFICIAL_SUFFIXES = (
    ".gov", ".mil", ".edu",
    ".gov.uk", ".gov.au", ".gov.cn", ".gov.in", ".gov.sg", ".gov.ca",
    ".ac.uk", ".ac.jp", ".ac.cn", ".ac.in", ".ac.kr",
    ".edu.cn", ".edu.au", ".edu.hk", ".edu.sg",
)

# Seed whitelist -> "trusted", templated reason, no LLM call. Bootstraps the
# Redis cache on first sight; hand-picked, extend freely.
_SEED_TRUSTED_DOMAINS = {
    "wikipedia.org", "wikimedia.org",
    "google.com", "apple.com", "microsoft.com", "amazon.com",
    "github.com", "stackoverflow.com", "mozilla.org",
    "who.int", "un.org",
    "nytimes.com", "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "theguardian.com", "npr.org", "wsj.com", "bloomberg.com", "economist.com",
    "nature.com", "sciencedirect.com", "ieee.org",
}

# User-generated-content platforms: never resolved or cached at the domain
# level (a single reputable-looking domain hosts everything from expert
# answers to spam) — always deferred to the per-source LLM call, which still
# assigns them their own reason same as any other LLM-judged source.
_PLATFORM_DOMAINS = {
    "reddit.com", "x.com", "twitter.com", "facebook.com", "instagram.com",
    "threads.net", "tiktok.com", "youtube.com", "linkedin.com", "pinterest.com",
    "tumblr.com", "medium.com", "substack.com", "blogspot.com", "wordpress.com",
    "github.io", "notion.site", "quora.com",
}

_SNIPPET_CHARS = 50


def _domain_of(url: str) -> str:
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _matches(host: str, domain_set: set[str]) -> str | None:
    """Return the matched base domain if `host` equals it or is a subdomain of it."""
    for base in domain_set:
        if host == base or host.endswith("." + base):
            return base
    return None


def _is_official(host: str) -> bool:
    return any(host == s.lstrip(".") or host.endswith(s) for s in _OFFICIAL_SUFFIXES)


def _official_credibility(host: str) -> Credibility:
    return {
        "label": "official",
        "reason": f"{host} is a government, military, or accredited educational institution domain.",
    }


def _seed_credibility(domain: str) -> Credibility:
    return {
        "label": "trusted",
        "reason": f"{domain} is on our curated list of well-established, editorially reliable sources.",
    }


_UNKNOWN_NO_URL: Credibility = {
    "label": "unknown",
    "reason": "No URL was available for this source, so it couldn't be classified.",
}
_UNKNOWN_LLM_FAILED: Credibility = {
    "label": "unknown",
    "reason": "The classifier could not confidently determine this source's credibility.",
}


class _CredibilityItem(BaseModel):
    index: int = Field(description="The candidate's index, as given in the prompt.")
    label: CredibilityLabel = Field(
        description="The credibility classification for this source."
    )
    reason: str = Field(
        description="One concise, plain-language sentence explaining the judgment "
        "for this specific source — not a generic definition of the label."
    )


class _CredibilityResult(BaseModel):
    items: list[_CredibilityItem]


# `method="json_mode"`, matching source_rerank.py: gpt-oss-20b's tool-calling
# on Groq auto-camelCases the registered class name and fails Groq's strict
# tool-name validation. json_mode sidesteps that entirely.
_classifier_model = credibility_llm.with_structured_output(
    _CredibilityResult, method="json_mode"
)

_SYSTEM_PROMPT = """You are rating how much an AI assistant should trust each web source before citing it.

You will be given the user's query/topic (if known) and a numbered list of candidate sources (domain, title, and a short snippet). For EACH candidate, assign exactly one label AND a one-sentence reason:

- "official": a government, military, or accredited educational/institutional domain.
- "trusted": a well-established, editorially rigorous, widely recognized reliable source (major news outlets, encyclopedic references, standards bodies, established organizations' own documentation) — but not a government/official body.
- "first_party": the page belongs to the exact entity, person, product, or organization the user's query is asking about (e.g. the user asks about a company and the source is that company's own domain). Judge this from the query, not general trustworthiness.
- "social_media": a social media, forum, or user-generated-content platform post — credibility depends on the specific poster/account, not the platform.
- "junk": low-quality, spam, content-farm, clickbait, or otherwise unreliable.
- "unknown": not enough information to confidently classify.

`reason`: one concise, plain-language sentence explaining your judgment for THIS SPECIFIC source (not a generic definition of the label) — e.g. "This is Reuters' own news domain, a long-established wire service with editorial standards" rather than just "It's a trusted news source."

Return ONLY a JSON object of the form {"items": [{"index": int, "label": str, "reason": str}]}, one item per candidate, indices matching the input."""


async def classify_sources(items: list[dict], query: str | None) -> list[dict]:
    """Attach a `credibility` dict (`{"label", "reason"}`) to each source dict.

    Resolves each item through regex -> Redis/seed whitelist -> a single
    batched gpt-oss-20b call for whatever's left. Never raises — anything
    that can't be resolved (empty URL, LLM failure) is labeled "unknown"
    with a generic reason. Returns new dicts; `items` is not mutated. Order
    is preserved.
    """
    out: list[dict | None] = [None] * len(items)
    host_of: dict[int, str] = {}
    cache_key_of: dict[int, str | None] = {}
    seed_domain_of: dict[int, str] = {}
    cache_lookup: dict[str, list[int]] = {}
    pending_llm: list[int] = []

    for i, item in enumerate(items):
        url = str(item.get("url") or "").strip()
        host = _domain_of(url)
        host_of[i] = host

        if not host:
            out[i] = {**item, "credibility": _UNKNOWN_NO_URL}
            continue

        if _is_official(host):
            out[i] = {**item, "credibility": _official_credibility(host)}
            continue

        if _matches(host, _PLATFORM_DOMAINS) is not None:
            cache_key_of[i] = None
            pending_llm.append(i)
            continue

        seed_match = _matches(host, _SEED_TRUSTED_DOMAINS)
        cache_key = seed_match or host
        cache_key_of[i] = cache_key
        if seed_match:
            seed_domain_of[i] = seed_match
        cache_lookup.setdefault(cache_key, []).append(i)

    to_cache_trusted: dict[str, str] = {}

    if cache_lookup:
        cached = await credibility_redis.get_many(list(cache_lookup.keys()))
        for cache_key, indices in cache_lookup.items():
            credibility = _decode_cached(cached.get(cache_key))
            if credibility is not None:
                for i in indices:
                    out[i] = {**items[i], "credibility": credibility}
            elif cache_key in seed_domain_of.values():
                credibility = _seed_credibility(cache_key)
                for i in indices:
                    out[i] = {**items[i], "credibility": credibility}
                to_cache_trusted[cache_key] = json.dumps(credibility)
            else:
                pending_llm.extend(indices)

    llm_labels: dict[int, Credibility] = {}
    if pending_llm:
        llm_labels = await _classify_via_llm(items, pending_llm, host_of, query)

    for i in pending_llm:
        credibility = llm_labels.get(i, _UNKNOWN_LLM_FAILED)
        out[i] = {**items[i], "credibility": credibility}
        cache_key = cache_key_of.get(i)
        if cache_key and credibility["label"] == "trusted":
            to_cache_trusted[cache_key] = json.dumps(credibility)

    if to_cache_trusted:
        await credibility_redis.set_many(to_cache_trusted, TTL_TRUSTED)

    return [
        out[i] if out[i] is not None else {**items[i], "credibility": _UNKNOWN_LLM_FAILED}
        for i in range(len(items))
    ]


def _decode_cached(raw: str | None) -> Credibility | None:
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"[source_credibility] discarding malformed cache entry: {raw!r}")
        return None
    if isinstance(decoded, dict) and decoded.get("label") and decoded.get("reason"):
        return decoded  # type: ignore[return-value]
    return None


async def _classify_via_llm(
    items: list[dict],
    indices: list[int],
    host_of: dict[int, str],
    query: str | None,
) -> dict[int, Credibility]:
    listing_lines = []
    for pos, i in enumerate(indices):
        item = items[i]
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("content") or "").strip()[:_SNIPPET_CHARS]
        listing_lines.append(
            f"[{pos}] domain: {host_of[i]} | title: {title} | snippet: {snippet}"
        )
    listing = "\n".join(listing_lines)
    topic = query.strip() if query else "(not given)"
    messages = [
        ("system", _SYSTEM_PROMPT),
        ("human", f"User query/topic:\n{topic}\n\nCandidates:\n{listing}"),
    ]

    result = None
    for attempt in range(2):
        try:
            with tracing_context(project_name="source_credibility"):
                result = await _classifier_model.ainvoke(messages)
            break
        except Exception as exc:
            logger.warning(f"[source_credibility] classify attempt {attempt} failed: {exc}")
    if result is None:
        return {}

    labels: dict[int, Credibility] = {}
    for entry in result.items:
        if entry.index < 0 or entry.index >= len(indices):
            continue
        labels[indices[entry.index]] = {
            "label": entry.label,
            "reason": entry.reason.strip(),
        }
    return labels
