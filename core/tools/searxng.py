"""Thin HTTP client for our self-hosted SearXNG instance.

SearXNG is a metasearch engine: it fans a query out to real engines (Google,
Brave, DuckDuckGo, OpenStreetMap, …) and merges the results. This module is
the single place that talks to it, and it replaces the Google Serper API
everywhere it used to be used — web search, image search, and the entity
card that used to read Serper's `knowledgeGraph`.

The JSON API (https://docs.searxng.org/dev/engines/json_engine.html) is a GET
to ``/search`` with ``format=json``; the body is::

    {"query": ..., "results": [...], "answers": [...], "infoboxes": [...],
     "suggestions": [...], "unresponsive_engines": [...]}

Every result carries the fields of the template that renders it, so the shape
varies by category. The normalizers below flatten each category into the
canonical shape the tool adapters expect (see ``core/tools/adapters.py``).
"""

import os

import httpx

# Public instance is the default so the tools work without any extra env
# wiring; override per-environment if we ever move it.
SEARXNG_BASE_URL = os.environ.get(
    "SEARXNG_BASE_URL",
    "https://searxng-railway-production-3785.up.railway.app",
).rstrip("/")

_DEFAULT_TIMEOUT = 10.0

# SearXNG's bot limiter looks at request headers; a client with no
# User-Agent is one of the things it filters out.
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; omni/1.0; +https://omni.chat)",
}

# SearXNG's own time filter. Google's `tbs=qdr:*` syntax is still accepted on
# the way in (see `normalize_time_range`) because prompts and a fine-tuned
# model may still emit it, but "past hour" has no SearXNG equivalent and
# degrades to "past day" — the nearest window that actually exists.
TIME_RANGES = ("day", "week", "month", "year")

_LEGACY_TBS_TO_TIME_RANGE = {
    "qdr:h": "day",
    "qdr:d": "day",
    "qdr:w": "week",
    "qdr:m": "month",
    "qdr:y": "year",
}


def normalize_time_range(time_range: str | None) -> str | None:
    """Coerce a caller-supplied time filter to one SearXNG understands.

    Returns None for anything unrecognised rather than raising: a bad filter
    should widen the search, never fail it.
    """
    if not time_range:
        return None
    value = str(time_range).strip().lower()
    if value in TIME_RANGES:
        return value
    return _LEGACY_TBS_TO_TIME_RANGE.get(value)


def searxng_search(
    query: str,
    *,
    categories: str | None = None,
    engines: str | None = None,
    time_range: str | None = None,
    language: str | None = None,
    pageno: int = 1,
    safesearch: int = 0,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict:
    """Run one SearXNG query and return the raw JSON body.

    Raises on transport/HTTP errors — callers decide whether a failed search
    is fatal or degrades to an empty result.
    """
    params: dict[str, str | int] = {
        "q": query,
        "format": "json",
        "pageno": pageno,
        "safesearch": safesearch,
    }
    if categories:
        params["categories"] = categories
    if engines:
        params["engines"] = engines
    normalized_range = normalize_time_range(time_range)
    if normalized_range:
        params["time_range"] = normalized_range
    if language:
        params["language"] = language

    resp = httpx.get(
        f"{SEARXNG_BASE_URL}/search",
        params=params,
        headers=_HEADERS,
        timeout=timeout,
        follow_redirects=True,
    )
    resp.raise_for_status()
    body = resp.json()
    return body if isinstance(body, dict) else {}


# ── Web search ──────────────────────────────────────────────────────────────


def _clean(value) -> str:
    return str(value or "").strip()


def search_web(
    query: str, k: int = 5, time_range: str | None = None
) -> list[dict]:
    """Normalized general web results: ``[{"title", "url", "content"}, …]``.

    `answers` and `infoboxes` are folded in ahead of the organic results, the
    same way Serper's answerBox/knowledgeGraph used to be — they're direct
    answers to the query and belong at the top of what the agent reads.
    """
    raw = searxng_search(query, time_range=time_range)

    normalized: list[dict] = []

    for answer in raw.get("answers") or []:
        # Newer SearXNG returns answer objects; older ones return bare strings.
        if isinstance(answer, dict):
            text = _clean(answer.get("answer") or answer.get("content"))
            url = _clean(answer.get("url"))
        else:
            text = _clean(answer)
            url = ""
        if text:
            normalized.append(
                {
                    "title": "Direct answer",
                    "url": url,
                    "content": text,
                }
            )

    for box in raw.get("infoboxes") or []:
        if not isinstance(box, dict):
            continue
        content = _clean(box.get("content"))
        if not content:
            continue
        url = _clean(box.get("url"))
        if not url:
            urls = box.get("urls") or []
            if urls and isinstance(urls[0], dict):
                url = _clean(urls[0].get("url"))
        normalized.append(
            {
                "title": _clean(box.get("infobox")) or _clean(box.get("title")),
                "url": url,
                "content": "Knowledge panel: " + content,
            }
        )

    for item in raw.get("results") or []:
        if not isinstance(item, dict):
            continue
        title = _clean(item.get("title"))
        url = _clean(item.get("url"))
        content = _clean(item.get("content"))
        if title or url or content:
            entry = {"title": title, "url": url, "content": content}
            published = _clean(item.get("publishedDate"))
            if published:
                entry["published_date"] = published
            normalized.append(entry)

    return normalized[:k]


# ── Image search ────────────────────────────────────────────────────────────

# Picking "the first image result" straight off SearXNG does not work: it
# merges ~40 engines and ranks by cross-engine agreement, so a clipart glyph
# or a Pinterest re-upload routinely outranks the photograph. Callers here
# want one usable photo, so results are filtered to raster images of real
# size and then ordered by how much the source engine can be trusted to
# return a photo rather than an illustration or a meme.
_PREFERRED_IMAGE_ENGINES = (
    "google cse images",
    "google images",
    "wikicommons.images",
    "unsplash",
    "pexels",
    "duckduckgo images",
    "brave.images",
    "bing images",
    "qwant images",
    "openverse",
    "flickr",
)

# Anything under this in either dimension is a thumbnail or an icon, not a
# picture worth putting on a report cover or an entity card.
_MIN_IMAGE_EDGE = 600

# Hosts that mostly serve re-uploads: the image is usually a screenshot,
# collage or animation rather than the thing that was searched for.
_IMAGE_HOST_DENYLIST = ("i.pinimg.com", "lookaside.fbsbx.com")


def _image_edges(resolution) -> tuple[int, int] | None:
    """Parse SearXNG's free-form resolution string ("2500x1667", "3011 × 2866")."""
    digits = "".join(ch if ch.isdigit() else " " for ch in str(resolution or ""))
    parts = digits.split()
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _image_candidate(item: dict) -> tuple[str, str] | None:
    """Return (img_src, source_link) if this result is a usable photo."""
    if not isinstance(item, dict):
        return None
    img_src = _clean(item.get("img_src"))
    # Some engines return protocol-relative URLs, which are broken anywhere
    # we re-host or embed them.
    if not img_src.startswith("http"):
        return None
    lowered = img_src.lower().split("?")[0]
    if lowered.endswith((".svg", ".gif")):
        return None
    if _clean(item.get("img_format")).upper() in {"SVG", "GIF"}:
        return None
    if any(host in img_src for host in _IMAGE_HOST_DENYLIST):
        return None
    edges = _image_edges(item.get("resolution"))
    if edges and min(edges) < _MIN_IMAGE_EDGE:
        return None
    return img_src, _clean(item.get("url"))


def search_image(query: str) -> tuple[str, str] | None:
    """Return ``(image_url, source_link)`` for the best photo hit, or None."""
    raw = searxng_search(query, categories="images")

    best: tuple[int, tuple[str, str]] | None = None
    for item in raw.get("results") or []:
        candidate = _image_candidate(item)
        if candidate is None:
            continue
        engine = _clean(item.get("engine")).lower()
        try:
            rank = _PREFERRED_IMAGE_ENGINES.index(engine)
        except ValueError:
            # Unknown engine — usable, but only if nothing better shows up.
            rank = len(_PREFERRED_IMAGE_ENGINES)
        if best is None or rank < best[0]:
            best = (rank, candidate)
        if rank == 0:
            break

    return best[1] if best else None


# ── Infobox (the knowledge-graph replacement) ───────────────────────────────


def fetch_infobox(query: str) -> dict | None:
    """Return the first Wikidata/Wikipedia infobox for ``query``, or None.

    This is SearXNG's stand-in for Serper's `knowledgeGraph`: a structured
    entity summary with a description, an image and typed attributes.
    """
    raw = searxng_search(query)
    for box in raw.get("infoboxes") or []:
        if isinstance(box, dict) and (_clean(box.get("infobox")) or _clean(box.get("content"))):
            return box
    return None
