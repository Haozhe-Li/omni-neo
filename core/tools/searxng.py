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
import time

import httpx

# Public instance is the default so the tools work without any extra env
# wiring; override per-environment if we ever move it.
SEARXNG_BASE_URL = os.environ.get(
    "SEARXNG_BASE_URL",
    "https://searxng-railway-production-3785.up.railway.app",
).rstrip("/")

_DEFAULT_TIMEOUT = 10.0

# Which engines `search_web` asks for, by name, in one request. Sent as
# SearXNG's `engines` query param — a per-request selection that needs no
# change to the instance's settings.yml.
#
# Naming them beats relying on the instance's default category. That default
# was four web engines plus six lookup plugins, and on 2026-09-07 all four of
# the real ones were suspended at once (brave/google cse rate-limited,
# startpage CAPTCHA'd, duckduckgo timing out) — every search in the product
# returned nothing, with no redundancy to fall back on. This list is the whole
# `web` category instead, so a single engine dying costs coverage rather than
# service.
#
# Kept deliberately wide, including engines currently scoring badly on the
# instance's own reliability panel: they recover, and the cost of carrying a
# dead one is a name in `unresponsive_engines`, while the cost of a short list
# is another total outage. Quality is filtered downstream — `_attribute` in
# core/tools/adapters.py credibility-classifies every result and drops the junk
# before the agent reads it, which is what handles the two engines measured
# returning off-query or spam results here (bing answered a vaccines query with
# "Integer Calculator"; qwant returns SEO-farm domains on Chinese queries).
_DEFAULT_ENGINES = (
    "bing",
    "brave",
    "duckduckgo",
    "google",
    "google cse",
    "mojeek",
    "qwant",
    "startpage",
    "yahoo",
)
SEARXNG_ENGINES = os.environ.get("SEARXNG_ENGINES", ",".join(_DEFAULT_ENGINES)).strip()

# Statuses worth trying again. 429/503 are the bot limiter; 400/422 are what
# the instance returns when the engines it queried all failed, which is a
# property of that moment rather than of the query.
_RETRY_STATUS = frozenset({400, 422, 429, 500, 502, 503, 504})


class SearxngUnavailable(RuntimeError):
    """Every upstream engine failed, so the empty body means nothing.

    SearXNG does not fail the request when the engines it fans out to do: it
    answers 200 with `results: []` and names the casualties in
    `unresponsive_engines`. Read naively that is indistinguishable from a query
    with genuinely no matches, and the difference matters enormously — the
    caller tells the agent "no results found, please change your query", so the
    agent rewrites a perfectly good query, searches again into the same outage,
    and burns its budget without ever being able to tell the user the truth.

    Raised so the retry loop treats it like any other transport failure, and so
    a persistent outage surfaces to `core/tools/adapters.py` as an exception,
    where it degrades to "search unavailable — say you could not verify this"
    instead of a lie about the query.
    """
_MAX_ATTEMPTS = 3
# Fewer attempts when every engine is down. An HTTP error or a timeout is often
# one bad request and worth three tries; engines suspended for rate limiting or
# sitting behind a CAPTCHA will not recover inside a 4-second backoff, and at
# three attempts each search costs ~11s of pure waiting — several of those in
# one turn is a minute of silence before the agent can even say it failed.
_MAX_ATTEMPTS_ALL_ENGINES_DOWN = 2
_RETRY_BACKOFF = 1.5

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


def _warn_if_engines_ignored(dead_engines: list) -> None:
    """Catch a mistyped `SEARXNG_ENGINES` before it looks like an outage.

    SearXNG does not reject an engine name it does not know — it quietly falls
    back to the whole general category instead. Verified: `engines=` with a
    nonsense value came back reporting brave/duckduckgo/google cse/startpage as
    unresponsive, which are exactly the defaults and none of them was asked
    for. So a typo produces the precise failure the setting exists to escape,
    and looks identical to "the engine I picked has no results".

    An engine we did not ask for appearing in the casualty list is the tell.
    """
    if not SEARXNG_ENGINES or not dead_engines:
        return
    asked = {e.strip().lower() for e in SEARXNG_ENGINES.split(",") if e.strip()}
    reported = {
        (d[0] if isinstance(d, (list, tuple)) else str(d)).strip().lower()
        for d in dead_engines
    }
    unasked = reported - asked
    if unasked:
        print(
            f"[searxng] SEARXNG_ENGINES={SEARXNG_ENGINES!r} was ignored — the "
            f"instance fell back to its default engines ({', '.join(sorted(unasked))}). "
            "Check the names against the instance's /config."
        )


def _raise_if_all_engines_down(body: dict) -> None:
    """Reject an empty body that is empty *because the engines failed*.

    Only when the response carries nothing usable at all AND names unresponsive
    engines. A query with genuinely no matches comes back empty with no
    casualties, and that is a real answer the caller should pass through as
    "no results" — the distinction is the whole point of this function.
    """
    if body.get("results") or body.get("answers") or body.get("infoboxes"):
        return
    dead = body.get("unresponsive_engines") or []
    if not dead:
        return
    names = ", ".join(
        f"{d[0]} ({d[1]})" if isinstance(d, (list, tuple)) and len(d) > 1 else str(d)
        for d in dead[:6]
    )
    raise SearxngUnavailable(f"no results and every engine failed: {names}")


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

    Retries a handful of times before giving up. SearXNG is a metasearch
    front-end over engines that rate-limit it, and it answers a request it
    could not serve with 4xx as readily as 5xx — 429 and 503 from the limiter,
    but also 400/422 when the upstream engines it fanned out to all came back
    empty or malformed. Those are transient in the ordinary case: the same
    query a second later fans out to a different set of responsive engines and
    succeeds. Without a retry each one costs a whole turn its evidence, and
    during data collection it silently produces a trajectory that answers with
    no sources at all.

    Raises after the last attempt — callers decide whether a failed search is
    fatal or degrades to an empty result.
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

    last: Exception | None = None
    attempts = _MAX_ATTEMPTS
    for attempt in range(attempts):
        try:
            resp = httpx.get(
                f"{SEARXNG_BASE_URL}/search",
                params=params,
                headers=_HEADERS,
                timeout=timeout,
                follow_redirects=True,
            )
            resp.raise_for_status()
            body = resp.json()
            if not isinstance(body, dict):
                return {}
            _raise_if_all_engines_down(body)
            return body
        except httpx.HTTPStatusError as exc:
            last = exc
            if exc.response.status_code not in _RETRY_STATUS:
                raise
        except (httpx.TransportError, ValueError, SearxngUnavailable) as exc:
            # TransportError covers timeouts and connection resets; ValueError
            # is a body that did not parse as JSON, which the instance returns
            # as an HTML error page under load.
            last = exc
            if isinstance(exc, SearxngUnavailable):
                attempts = min(attempts, _MAX_ATTEMPTS_ALL_ENGINES_DOWN)
        if attempt < attempts - 1:
            time.sleep(_RETRY_BACKOFF * (2 ** attempt))
    raise last if last else RuntimeError("searxng: exhausted retries with no error")


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
    raw = searxng_search(query, time_range=time_range, engines=SEARXNG_ENGINES or None)
    dead_engines = raw.get("unresponsive_engines") or []
    _warn_if_engines_ignored(dead_engines)

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

    # The outage check has to run *here*, on what survived normalization, not on
    # the raw body. `searxng_search` already rejects a body with nothing in it
    # at all, but a body can carry an infobox that this function then drops (no
    # url, no content) — "how vaccines work" does exactly that while all four
    # web engines are suspended. Judged on the raw body it looks like a served
    # query; judged on the output it is an outage, and only the caller's view
    # is the honest one.
    if not normalized and dead_engines:
        names = ", ".join(
            f"{d[0]} ({d[1]})" if isinstance(d, (list, tuple)) and len(d) > 1 else str(d)
            for d in dead_engines[:6]
        )
        raise SearxngUnavailable(f"nothing usable and every engine failed: {names}")

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
