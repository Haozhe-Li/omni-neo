"""Generate a cover photo for a published report.

Reports are LLM-written and never come with a real cover image. Given just
the report title, this module:

1. Rewrites the title into a short, concrete image-search query with a fast
   LLM (titles are long/abstract — searching them verbatim mostly returns
   screenshots and infographics, not usable photos).
2. Runs that query through Google Image Search (Serper) and takes the first
   hit — same "search once, take the first result" pattern already used for
   entity widgets, see ``core/widget_predictor.py::_fetch_entity_image``.
3. Re-hosts the image on our own R2/CDN bucket, since the Serper result is a
   hotlink to a random third-party page that can disappear at any time,
   which would silently break a cover that's meant to stay up indefinitely.

Called synchronously from the frontend's publish flow, once, the first time
a report is published (never regenerated on republish). Every failure mode
(bad query, no image results, download/upload failure) degrades to
``{"image_url": None, "source_link": None}`` — the frontend then falls back
to its abstract placeholder cover instead of blocking or failing publish.
"""

from __future__ import annotations

import logging
import os
import uuid

import boto3
import httpx
from langchain_community.utilities import GoogleSerperAPIWrapper
from langsmith import tracing_context

from core.llm import generate_cover_llm

logger = logging.getLogger(__name__)

_COVER_QUERY_SYSTEM_PROMPT = """
You turn a research report title into a short image-search query that will find a good, concrete cover photo.

Rules:
- Output ONLY the search query, nothing else (no quotes, no punctuation, no explanation).
- Always in English, 2-5 words — translate non-English titles.
- Extract the single most concrete, photographable subject behind the title (a physical object, place, industry scene, or action). Drop years, report-speak ("analysis", "assessment", "trends", "the state of"), and abstract nouns that have no photo (e.g. "risk", "impact", "outlook").
- If the literal topic isn't visual, substitute a generic real-world scene that represents it (e.g. interest-rate policy -> "federal reserve building"; chip supply chains -> "semiconductor factory").
- Never include the words "report", "analysis", "chart", "graph", or "diagram".

Examples:
Title: Global Semiconductor Supply Chain Risk Assessment 2026
Query: semiconductor factory

Title: The State of Remote Work in 2026
Query: person working from home

Title: Comparative Analysis of Renewable Energy Storage Technologies
Query: solar panels battery storage

Title: 比特币价格会涨吗
Query: bitcoin coins

Now generate the query for the given title.
"""

_COVER_BUCKET = os.getenv("S3_BUCKET_NAME", "omni")
_CDN_BASE = "https://cdn.omniknows.xyz/public"

_EXTENSION_MAP = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}

_DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}

s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("S3_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    region_name="auto",
)


def _generate_cover_query(title: str) -> str:
    messages = [
        ("system", _COVER_QUERY_SYSTEM_PROMPT),
        ("human", f"The title is: {title}"),
    ]
    with tracing_context(project_name="generate-cover"):
        res = generate_cover_llm.invoke(messages).content
    return str(res).strip().strip('"')


def _search_first_image(query: str) -> tuple[str, str] | None:
    """Return (imageUrl, sourceLink) for the first Serper image result, or None."""
    try:
        img_search = GoogleSerperAPIWrapper(k=1, type="images")
        raw = img_search.results(query)
        images = raw.get("images") or []
        if not images:
            return None
        first = images[0]
        image_url = first.get("imageUrl") or ""
        if not image_url:
            return None
        return image_url, first.get("link") or ""
    except Exception as exc:
        logger.warning(f"[generate_cover] image search failed for {query!r}: {exc}")
        return None


def _migrate_to_r2(image_url: str) -> str | None:
    """Download the hotlinked image and re-host it on our own CDN so the cover
    survives even if the source page it was scraped from disappears."""
    try:
        with httpx.Client(timeout=8, follow_redirects=True) as client:
            resp = client.get(image_url, headers=_DOWNLOAD_HEADERS)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
        if not content_type.startswith("image/"):
            logger.warning(f"[generate_cover] not an image (content-type={content_type}): {image_url}")
            return None

        ext = _EXTENSION_MAP.get(content_type, content_type.split("/")[-1] or "jpg")
        filename = f"cover-{uuid.uuid4().hex}.{ext}"

        s3_client.put_object(
            Bucket=_COVER_BUCKET,
            Key=f"public/{filename}",
            Body=resp.content,
            ContentType=content_type,
        )
        return f"{_CDN_BASE}/{filename}"
    except Exception as exc:
        logger.warning(f"[generate_cover] failed to migrate cover image {image_url}: {exc}")
        return None


def generate_cover(title: str) -> dict:
    """Given a report title, return {"image_url": str | None, "source_link": str | None}.

    Never raises — every step is best-effort, so a bad title, a flaky search,
    or an R2 hiccup just falls back to an empty result instead of failing
    whatever publish flow called this synchronously.
    """
    try:
        query = _generate_cover_query(title)
        logger.info(f"[generate_cover] title={title!r} -> query={query!r}")

        hit = _search_first_image(query)
        if not hit:
            return {"image_url": None, "source_link": None}
        raw_image_url, source_link = hit

        cdn_url = _migrate_to_r2(raw_image_url)
        if not cdn_url:
            return {"image_url": None, "source_link": None}

        return {"image_url": cdn_url, "source_link": source_link}
    except Exception as exc:
        logger.warning(f"[generate_cover] failed for title={title!r}: {exc}")
        return {"image_url": None, "source_link": None}
