"""Credibility classification for web sources (`web_search` / `fetch_url`).

Every source gets a `credibility` dict — `{"label": ..., "reason": ...}` — not
just a bare label. `reason` is always a one-sentence, human-readable
explanation; where it comes from depends on the layer that resolved the
source, cheapest first:

0. `_HARDCODED_BYPASS_DOMAINS` — an explicit set of domains we own outright,
   hardcoded to "trusted" in-process. No cache read/write, no LLM, checked
   before anything else. Distinct from (1)/(2) below: those are general
   patterns/whitelists resolved through Redis, this is a fixed, source-owned
   exemption that must never depend on cache state.
1. Regex on the hostname — unambiguous government/military/educational
   domains resolve to "official" for free, no cache, no LLM, reason is a
   templated sentence.
2. A Redis-backed domain whitelist (`redis_credibility.py`), bootstrapped by
   a small hardcoded seed list (wikipedia/google/apple/...) -> "trusted"
   with a templated reason. Lookups are batched with one MGET per call.
3. Whatever's left after (0)-(2) — including user-generated-content
   platforms (reddit, x.com, medium, ...), which are deliberately never
   resolved by (1)/(2) since quality varies wildly page-to-page — goes
   through a single batched gpt-oss-20b call (title + url + a short snippet
   + the user's query/topic, for judging "first_party"). The model produces
   both the label and its own one-sentence reason. Only "trusted" and
   "arguable" verdicts get written back into the Layer 2 cache (as JSON,
   `{label, reason}` together) — both are properties of the *domain*, not
   the query, so they're safe to reuse. "official" is left to the regex
   layer, "first_party" is a property of (domain, query) not of the domain
   alone, "social_media" and "unknown" carry no reusable signal, and "junk"
   is deliberately never cached: a page being junk for *this* query doesn't
   mean the whole domain is junk (could just be one marketing page on an
   otherwise fine site) — so every hit gets a fresh LLM judgment instead of
   inheriting a stale domain-wide verdict.

"arguable" sources (layer 2 or 3) are never dropped the way "junk" is —
they're surfaced to the agent with their `reason` attached so it can weigh
and attribute the content instead of citing it as settled fact. See
`_CREDIBILITY_RANK` in core/tools/adapters.py for the ranking/visibility
rule, and this module's `_SYSTEM_PROMPT` for what does and doesn't qualify
for the label (a documented reliability record, never a viewpoint).
"""
from __future__ import annotations

import json
import logging
from typing import Literal, TypedDict
from urllib.parse import urlsplit

from langsmith import tracing_context
from pydantic import BaseModel, Field

from core.llm import credibility_llm
from core.utils.redis_credibility import TTL_DOMAIN_VERDICT, credibility_redis

logger = logging.getLogger(__name__)

CredibilityLabel = Literal[
    "official", "trusted", "first_party", "social_media", "arguable", "junk", "unknown"
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

# Hardcoded bypass: these domains (and subdomains) always resolve to
# "trusted" with zero lookups — no Redis read/write, no LLM call. Checked
# before every other layer, same spirit as `_OFFICIAL_SUFFIXES` but for a
# domain we own outright rather than a generic government/edu pattern.
_HARDCODED_BYPASS_DOMAINS = {
    "omniknows.xyz",
}

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


def _bypass_credibility(host: str) -> Credibility:
    return {
        "label": "trusted",
        "reason": f"{host} is an owned/first-party domain, exempted from credibility classification.",
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
- "arguable": the domain has a documented, independently verifiable record of publishing fabricated, deceptively sourced, or manipulated content — e.g. flagged by independent fact-checking or media-reliability organizations, confirmed coordinated inauthentic behavior, a pattern of retracted or debunked stories. This is about a track record, not a viewpoint: a source being opinionated, partisan, or critical of a government, party, company, or public figure is NEVER by itself grounds for "arguable" — plenty of rigorous journalism is sharply critical of its subject. Only use this label when you can point to a specific, factual reliability problem, not a political alignment.
- "junk": low-quality, spam, content-farm, clickbait, or otherwise unreliable.
- "unknown": not enough information to confidently classify.

`reason`: one concise, plain-language sentence explaining your judgment for THIS SPECIFIC source (not a generic definition of the label) — e.g. "This is Reuters' own news domain, a long-established wire service with editorial standards" rather than just "It's a trusted news source." For "arguable", name the specific reliability problem (e.g. "rated low for factual reporting by independent media monitors after repeated fabricated stories"), never a political characterization.

Return ONLY a JSON object of the form {"items": [{"index": int, "label": str, "reason": str}]}, one item per candidate, indices matching the input."""


async def classify_single_url(url: str) -> Credibility:
    """Classify a single URL with no title/content/query context — used by
    POST /classify_url for raw markdown links the agent typed directly into
    an answer, which never went through `classify_sources`. Reuses the same
    tier order (bypass -> official regex -> seed/platform -> Redis cache ->
    LLM) and the same Redis keyspace/TTL, so a domain resolved by either path
    benefits the other. Never raises; unresolvable input returns
    `_UNKNOWN_NO_URL`.
    """
    host = _domain_of(url)
    if not host:
        return _UNKNOWN_NO_URL
    if _matches(host, _HARDCODED_BYPASS_DOMAINS) is not None:
        return _bypass_credibility(host)
    if _is_official(host):
        return _official_credibility(host)

    is_platform = _matches(host, _PLATFORM_DOMAINS) is not None
    seed_match = None if is_platform else _matches(host, _SEED_TRUSTED_DOMAINS)
    cache_key = seed_match or (None if is_platform else host)

    if cache_key:
        cached = _decode_cached((await credibility_redis.get_many([cache_key])).get(cache_key))
        if cached is not None:
            return cached
        if seed_match:
            credibility = _seed_credibility(seed_match)
            await credibility_redis.set_many({seed_match: json.dumps(credibility)}, TTL_DOMAIN_VERDICT)
            return credibility

    credibility = await _classify_url_via_llm(host, url)
    if cache_key and credibility["label"] in ("trusted", "arguable"):
        await credibility_redis.set_many({cache_key: json.dumps(credibility)}, TTL_DOMAIN_VERDICT)
    return credibility


class _UrlCredibilityResult(BaseModel):
    label: CredibilityLabel = Field(description="The credibility classification for this URL.")
    reason: str = Field(
        description="One concise, plain-language sentence explaining the judgment "
        "for this specific URL — not a generic definition of the label."
    )


# Separate model/prompt from `_classifier_model`/`_SYSTEM_PROMPT` on purpose:
# that one is framed around "does this source support the topic" and expects
# title/content/query, none of which exist here — a bare-URL judgment is a
# structurally different question (does the URL itself look suspicious).
_url_classifier_model = credibility_llm.with_structured_output(
    _UrlCredibilityResult, method="json_mode"
)

_URL_SYSTEM_PROMPT = """You are judging whether a bare URL looks safe to click, using ONLY the URL string — no page title, content, or query is available, so never guess at what the page is about.

Judge purely from structure:
- IP-literal hosts, or hosts using punycode/homoglyph lookalikes of a known brand.
- A domain that imitates a well-known brand via inserted words or a different TLD (e.g. "paypal-secure-login.com", "apple.com.verify-account.net").
- A known URL-shortener domain (bit.ly, tinyurl.com, t.co, goo.gl, is.gd, ...) — shorteners hide the real destination and are inherently unverifiable from the URL alone.
- Excessive/suspicious subdomain nesting combining a brand keyword with an unrelated domain.

Never assign "official", "trusted", or "first_party" — those require context (institutional recognition, editorial record, or matching what a user asked about) that a bare URL cannot establish; leave them to the deterministic checks that already ran before you were consulted.
Never assign "arguable" — that label means a documented, independently verifiable content-reliability track record, which a URL string cannot show.

Use "social_media" only if the domain itself matches a known UGC/social platform pattern. Use "junk" only when the URL exhibits one of the structural red flags above. Otherwise use "unknown" — the right default for an ordinary domain you don't recognize; do not stretch to "junk" without a concrete structural reason.

`reason`: one concise sentence about THIS URL specifically, naming the actual pattern you saw (or the absence of one). Never speculate about content.

Return ONLY a JSON object of the form {"label": str, "reason": str}."""


async def _classify_url_via_llm(host: str, url: str) -> Credibility:
    messages = [("system", _URL_SYSTEM_PROMPT), ("human", f"URL: {url}\nDomain: {host}")]
    for attempt in range(2):
        try:
            with tracing_context(project_name="source_credibility"):
                result = await _url_classifier_model.ainvoke(messages)
            label = result.label
            # Defense in depth: never let the LLM self-grant a no-friction
            # tier from URL text alone, regardless of what the prompt says —
            # passthrough must only ever come from the deterministic tiers
            # above (bypass/official-regex/seed whitelist).
            if label in ("official", "trusted", "first_party"):
                label = "unknown"
            return {"label": label, "reason": result.reason.strip()}
        except Exception as exc:
            logger.warning(f"[source_credibility] url-classify attempt {attempt} failed: {exc}")
    return _UNKNOWN_LLM_FAILED


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

        if _matches(host, _HARDCODED_BYPASS_DOMAINS) is not None:
            out[i] = {**item, "credibility": _bypass_credibility(host)}
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

    to_cache_domain: dict[str, str] = {}

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
                to_cache_domain[cache_key] = json.dumps(credibility)
            else:
                pending_llm.extend(indices)

    llm_labels: dict[int, Credibility] = {}
    if pending_llm:
        llm_labels = await _classify_via_llm(items, pending_llm, host_of, query)

    # "trusted"/"arguable" are properties of the domain, so they're worth
    # caching; every other label is either query-dependent (junk) or carries
    # no reusable signal (social_media, unknown) — see module docstring.
    for i in pending_llm:
        credibility = llm_labels.get(i, _UNKNOWN_LLM_FAILED)
        out[i] = {**items[i], "credibility": credibility}
        cache_key = cache_key_of.get(i)
        if cache_key and credibility["label"] in ("trusted", "arguable"):
            to_cache_domain[cache_key] = json.dumps(credibility)

    if to_cache_domain:
        await credibility_redis.set_many(to_cache_domain, TTL_DOMAIN_VERDICT)

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
