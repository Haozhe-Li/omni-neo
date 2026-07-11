"""Credibility classification for web sources (`google_search` / `load_web_page`).

Three layers, cheapest first:

1. Regex on the hostname — unambiguous government/military/educational
   domains resolve to "official" for free, no cache, no LLM.
2. A Redis-backed domain whitelist (`redis_credibility.py`), bootstrapped by
   a small hardcoded seed list (wikipedia/google/apple/...) -> "trusted".
   Lookups are batched with one MGET per call.
3. Whatever's left after (1) and (2) goes through a single batched
   gpt-oss-20b call (title + url + a short snippet + the user's query/topic,
   for judging "first_party"). Only "trusted" verdicts get written back into
   the Layer 2 cache — "official" is left to the regex layer, "first_party"
   is a property of (domain, query) not of the domain alone, "social_media"
   and "unknown" carry no reusable signal, and "junk" is deliberately never
   cached either: a page being junk for *this* query doesn't mean the whole
   domain is junk (could just be one marketing page on an otherwise fine
   site) — so every hit gets a fresh LLM judgment instead of inheriting a
   stale domain-wide verdict.

Domains that host arbitrary user-generated content (reddit, x.com, medium,
...) are exempt from caching in both directions: quality varies wildly
page-to-page, so they always go through the LLM and are never written back.
"""
from __future__ import annotations

import logging
from typing import Literal
from urllib.parse import urlsplit

from langsmith import tracing_context
from pydantic import BaseModel, Field

from core.llm import credibility_llm
from core.utils.redis_credibility import TTL_TRUSTED, credibility_redis

logger = logging.getLogger(__name__)

CredibilityLabel = Literal[
    "official", "trusted", "first_party", "social_media", "junk", "unknown"
]

# Suffixes that make a domain unambiguously official. Deterministic and
# free to check, so never cached. Extend as needed.
_OFFICIAL_SUFFIXES = (
    ".gov", ".mil", ".edu",
    ".gov.uk", ".gov.au", ".gov.cn", ".gov.in", ".gov.sg", ".gov.ca",
    ".ac.uk", ".ac.jp", ".ac.cn", ".ac.in", ".ac.kr",
    ".edu.cn", ".edu.au", ".edu.hk", ".edu.sg",
)

# Seed whitelist -> "trusted". Bootstraps the Redis cache on first sight;
# hand-picked, extend freely.
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
# answers to spam) — always deferred to the per-source LLM call.
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


class _CredibilityItem(BaseModel):
    index: int = Field(description="The candidate's index, as given in the prompt.")
    label: CredibilityLabel = Field(
        description="The credibility classification for this source."
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

You will be given the user's query/topic (if known) and a numbered list of candidate sources (domain, title, and a short snippet). For EACH candidate, assign exactly one label:

- "official": a government, military, or accredited educational/institutional domain.
- "trusted": a well-established, editorially rigorous, widely recognized reliable source (major news outlets, encyclopedic references, standards bodies, established organizations' own documentation) — but not a government/official body.
- "first_party": the page belongs to the exact entity, person, product, or organization the user's query is asking about (e.g. the user asks about a company and the source is that company's own domain). Judge this from the query, not general trustworthiness.
- "social_media": a social media, forum, or user-generated-content platform post — credibility depends on the specific poster/account, not the platform.
- "junk": low-quality, spam, content-farm, clickbait, or otherwise unreliable.
- "unknown": not enough information to confidently classify.

Return ONLY a JSON object of the form {"items": [{"index": int, "label": str}]}, one item per candidate, indices matching the input."""


async def classify_sources(items: list[dict], query: str | None) -> list[dict]:
    """Attach a `credibility` label to each source dict.

    Resolves each item through regex -> Redis/seed whitelist -> a single
    batched gpt-oss-20b call for whatever's left. Never raises — anything
    that can't be resolved (empty URL, LLM failure) is labeled "unknown".
    Returns new dicts; `items` is not mutated. Order is preserved.
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
            out[i] = {**item, "credibility": "unknown"}
            continue

        if _is_official(host):
            out[i] = {**item, "credibility": "official"}
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
            label = cached.get(cache_key)
            if label:
                for i in indices:
                    out[i] = {**items[i], "credibility": label}
            elif cache_key in seed_domain_of.values():
                for i in indices:
                    out[i] = {**items[i], "credibility": "trusted"}
                to_cache_trusted[cache_key] = "trusted"
            else:
                pending_llm.extend(indices)

    llm_labels: dict[int, CredibilityLabel] = {}
    if pending_llm:
        llm_labels = await _classify_via_llm(items, pending_llm, host_of, query)

    for i in pending_llm:
        label = llm_labels.get(i, "unknown")
        out[i] = {**items[i], "credibility": label}
        cache_key = cache_key_of.get(i)
        if cache_key and label == "trusted":
            to_cache_trusted[cache_key] = "trusted"

    if to_cache_trusted:
        await credibility_redis.set_many(to_cache_trusted, TTL_TRUSTED)

    return [
        out[i] if out[i] is not None else {**items[i], "credibility": "unknown"}
        for i in range(len(items))
    ]


async def _classify_via_llm(
    items: list[dict],
    indices: list[int],
    host_of: dict[int, str],
    query: str | None,
) -> dict[int, CredibilityLabel]:
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

    labels: dict[int, CredibilityLabel] = {}
    for entry in result.items:
        if entry.index < 0 or entry.index >= len(indices):
            continue
        labels[indices[entry.index]] = entry.label
    return labels
