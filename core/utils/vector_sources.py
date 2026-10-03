"""Chunked semantic index of every citation a thread has produced, backing
`/check_source`.

Storage is the self-hosted Qdrant; vectors come from the self-hosted embedding
service (dense: multilingual MiniLM, sparse: BM25). Retrieval is hybrid: a dense
and a sparse query are fused with reciprocal-rank fusion.

Indexing is fire-and-forget: `enqueue_source_indexing` is called from inside
`citations.register_citation`, which itself runs synchronously deep inside an
agent tool call — it must never block the token stream, so the actual chunk +
embed + upsert work runs on a small background thread pool and the caller never
waits on it.

All threads share one collection; isolation is a `thread_id` payload filter.
Chunk volume per thread is small (a couple hundred at most), so the search asks
for the top 100 per vector and does the fusion client-side.

Relevance gate. RRF scores are rank-based, so they cannot say "this is not
relevant". The gate is therefore the dense cosine similarity of a chunk to the
claim (`_MIN_SCORE`): a chunk is only a candidate if it clears it, and the
sparse ranking then helps order the candidates (exact terms, numbers, names).
Calibrated on real Omni sources against LLM-paraphrased claims: 0.40 keeps ~97% of
truly supporting sources (EN, ZH and cross-lingual) while a claim from an unrelated
thread clears it ~4% of the time. A miss ("no source found" for a supported claim)
costs more than a false candidate, which the LLM rerank step absorbs. Override with
VECTOR_MIN_SCORE.

Retention. A thread's chunks live exactly as long as its `threads_control` row:
explicit deletes remove them directly, expiry removes them for the threads it just
deleted, and `list_indexed_thread_ids` + `delete_threads_vectors` let
db_threads_control periodically reconcile — drop the chunks of any thread that no
longer exists — so nothing leaks even if one of those steps fails. (A plain
"older than 90 days" TTL would be wrong: a thread stays alive as long as it is
used, so its early chunks can be older than any fixed window.)
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import AsyncQdrantClient, QdrantClient, models

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 256
_CHUNK_OVERLAP = 32
_COLLECTION = os.getenv("VECTOR_COLLECTION", "omni_chunks")
_MIN_SCORE = float(os.getenv("VECTOR_MIN_SCORE", "0.40"))
_QUERY_LIMIT = 100
# A fetched page can be hundreds of thousands of characters; indexing all of it
# would put thousands of chunks through the shared embedding service for one
# source. Only the first _MAX_INDEX_CHARS of a source are indexed.
_MAX_INDEX_CHARS = int(os.getenv("VECTOR_MAX_INDEX_CHARS", "30000"))
# The embedding model reads at most 128 tokens (extra is truncated by the
# service); the service itself rejects a single text over 4000 characters.
_MAX_TEXT_CHARS = 4000
_EMBED_BATCH = 32
_RRF_K = 60

DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DENSE_DIM = 384

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP
)

# Sources are small (a few chunks), so a single request is dominated by round-trip
# overhead; the embedding service only reaches its ~50 chunks/s ceiling with a few
# requests in flight, and gains nothing from more (measured: 4 parallel = 32 parallel,
# while a deeper queue only delays interactive searches sharing the service).
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vector-index")

_lock = threading.Lock()
_ready = False
_sync_client: QdrantClient | None = None
_async_client: AsyncQdrantClient | None = None
_embed_sync: httpx.Client | None = None
_embed_async: httpx.AsyncClient | None = None


def _embed_url() -> str:
    return os.environ["EMBEDDING_SERVICE_URL"].rstrip("/")


def _qdrant_kwargs() -> dict:
    # port=None: honour the URL's own port (https://host => 443, not 6333).
    return {
        "url": os.environ["QDRANT_URL"],
        "port": None,
        "api_key": os.getenv("QDRANT_API_KEY"),
        "timeout": 30,
        "check_compatibility": False,
    }


def _get_sync_client() -> QdrantClient:
    global _sync_client
    with _lock:
        if _sync_client is None:
            _sync_client = QdrantClient(**_qdrant_kwargs())
        return _sync_client


def _get_async_client() -> AsyncQdrantClient:
    global _async_client
    with _lock:
        if _async_client is None:
            _async_client = AsyncQdrantClient(**_qdrant_kwargs())
        return _async_client


def _get_embed_sync() -> httpx.Client:
    global _embed_sync
    with _lock:
        if _embed_sync is None:
            _embed_sync = httpx.Client(timeout=httpx.Timeout(60, connect=5))
        return _embed_sync


def _get_embed_async() -> httpx.AsyncClient:
    global _embed_async
    with _lock:
        if _embed_async is None:
            _embed_async = httpx.AsyncClient(timeout=httpx.Timeout(30, connect=5))
        return _embed_async


def _ensure_ready() -> None:
    """Create the collection and payload indexes on first use (idempotent)."""
    global _ready
    if _ready:
        return
    client = _get_sync_client()
    with _lock:
        if _ready:
            return
        if not client.collection_exists(_COLLECTION):
            client.create_collection(
                collection_name=_COLLECTION,
                vectors_config={
                    "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
                },
                # Qdrant/bm25 vectors carry only term frequency; the IDF half is
                # applied by Qdrant at query time, which needs this modifier.
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
            logger.info(f"[vector_sources] created collection {_COLLECTION}")
        for field, schema in (
            ("thread_id", models.PayloadSchemaType.KEYWORD),
            ("turn", models.PayloadSchemaType.INTEGER),
            ("indexed_at", models.PayloadSchemaType.FLOAT),
        ):
            client.create_payload_index(_COLLECTION, field_name=field, field_schema=schema)
        _ready = True


# ── embedding service ───────────────────────────────────────────────────────


def _retry_delay(resp: httpx.Response | None, attempt: int) -> float:
    try:
        return min(float(resp.headers["Retry-After"]), 10.0)  # type: ignore[union-attr]
    except Exception:
        return 1.0 * (attempt + 1)


def _post_sync(path: str, payload: dict) -> dict:
    """POST to the embedding service, retrying 503 (queue full) and transport errors."""
    last: Exception | None = None
    for attempt in range(4):
        resp = None
        try:
            resp = _get_embed_sync().post(f"{_embed_url()}{path}", json=payload)
            if resp.status_code != 503:
                resp.raise_for_status()
                return resp.json()
            last = RuntimeError("embedding service busy (503)")
        except httpx.TransportError as e:
            last = e
        time.sleep(_retry_delay(resp, attempt))
    raise RuntimeError(f"embedding service failed: {last}")


async def _post_async(path: str, payload: dict) -> dict:
    last: Exception | None = None
    for attempt in range(3):
        resp = None
        try:
            resp = await _get_embed_async().post(f"{_embed_url()}{path}", json=payload)
            if resp.status_code != 503:
                resp.raise_for_status()
                return resp.json()
            last = RuntimeError("embedding service busy (503)")
        except httpx.TransportError as e:
            last = e
        await asyncio.sleep(_retry_delay(resp, attempt))
    raise RuntimeError(f"embedding service failed: {last}")


def _embed_docs_sync(texts: list[str]) -> tuple[list[list[float]], list[dict]]:
    dense: list[list[float]] = []
    sparse: list[dict] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        batch = [t[:_MAX_TEXT_CHARS] for t in texts[i : i + _EMBED_BATCH]]
        d = _post_sync("/embed/dense/text", {"texts": batch, "model": DENSE_MODEL})
        s = _post_sync("/embed/sparse", {"texts": batch, "is_query": False})
        if d["model"] != DENSE_MODEL or d["dim"] != DENSE_DIM:
            raise RuntimeError(f"unexpected dense model from service: {d['model']} {d['dim']}")
        dense += d["embeddings"]
        sparse += s["embeddings"]
    return dense, sparse


async def _embed_query(text: str) -> tuple[list[float], dict]:
    text = text[:_MAX_TEXT_CHARS]
    d, s = await asyncio.gather(
        _post_async("/embed/dense/text", {"texts": [text], "model": DENSE_MODEL}),
        _post_async("/embed/sparse", {"texts": [text], "is_query": True}),
    )
    if d["model"] != DENSE_MODEL or d["dim"] != DENSE_DIM:
        raise RuntimeError(f"unexpected dense model from service: {d['model']} {d['dim']}")
    return d["embeddings"][0], s["embeddings"][0]


async def embed_dense(texts: list[str]) -> list[list[float]]:
    """Dense vectors only, for callers that compare short strings rather than
    index them (core/intent_router.py). Same service, retries and model check as
    the indexing path; one request, so the caller batches."""
    d = await _post_async(
        "/embed/dense/text",
        {"texts": [t[:_MAX_TEXT_CHARS] for t in texts], "model": DENSE_MODEL},
    )
    if d["model"] != DENSE_MODEL or d["dim"] != DENSE_DIM:
        raise RuntimeError(f"unexpected dense model from service: {d['model']} {d['dim']}")
    return d["embeddings"]


# ── indexing ────────────────────────────────────────────────────────────────


def _point_id(thread_id: str, n: int, chunk_index: int) -> str:
    # Deterministic, so re-indexing the same source overwrites instead of duplicating.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{thread_id}:{n}:{chunk_index}"))


def _title_url_prefix(record: dict) -> str:
    """Title + URL (filename + URL for a document), one per line — prepended
    to every chunk before embedding so a claim that names the source itself
    ("according to the CDC...") can match on that, not just the body text.
    Empty when neither is set, so plain content chunks are unaffected."""
    title = (record.get("title") or "").strip()
    url = (record.get("url") or "").strip()
    return "\n".join(line for line in (title, url) if line)


def _index_source_sync(thread_id: str, record: dict) -> None:
    content = (record.get("content") or "").strip()[:_MAX_INDEX_CHARS]
    if not content:
        return
    n = record["n"]
    chunks = _splitter.split_text(content)
    if not chunks:
        return
    prefix = _title_url_prefix(record)
    # `text` carries the title/URL prefix baked in — it's both what gets embedded
    # *and* what `search_similar_chunks` later returns as `chunk`, so the same
    # prefixed text is what source_rerank.py's LLM call and the frontend's
    # highlight both see. No separate "embed-only" field: keeping it one field
    # guarantees any excerpt the rerank step copies is still a genuine
    # substring of what's displayed.
    texts = [f"{prefix}\n\n{c}" if prefix else c for c in chunks]

    _ensure_ready()
    dense, sparse = _embed_docs_sync(texts)
    now = time.time()
    points = [
        models.PointStruct(
            id=_point_id(thread_id, n, i),
            vector={
                "dense": dv,
                "sparse": models.SparseVector(indices=sv["indices"], values=sv["values"]),
            },
            payload={
                "thread_id": thread_id,
                "text": text,
                "n": n,
                "url": record.get("url", ""),
                "title": record.get("title", ""),
                "chunk_index": i,
                "turn": record.get("turn"),
                "credibility": record.get("credibility"),
                "indexed_at": now,
            },
        )
        for i, (text, dv, sv) in enumerate(zip(texts, dense, sparse))
    ]
    _get_sync_client().upsert(_COLLECTION, points=points)
    print(
        f"[vector_sources] indexed {len(points)} chunk(s) for thread={thread_id} "
        f"n={n} turn={record.get('turn')} title={record.get('title', '')!r}"
    )


def enqueue_source_indexing(thread_id: str, record: dict) -> None:
    """Fire-and-forget: chunk `record["content"]` and upsert to Qdrant.

    Never raises — indexing is best-effort and must not affect the calling
    tool or the token stream.
    """
    def _run() -> None:
        try:
            _index_source_sync(thread_id, record)
        except Exception as exc:
            logger.warning(f"[vector_sources] indexing failed for thread {thread_id}: {exc}")

    _executor.submit(_run)


# ── search ──────────────────────────────────────────────────────────────────


def _filter(thread_id: str, turn: int | None) -> models.Filter:
    must: list = [
        models.FieldCondition(key="thread_id", match=models.MatchValue(value=thread_id))
    ]
    if turn is not None:
        # Sources introduced at or before `turn`; a record with no turn is never excluded.
        must.append(
            models.Filter(
                should=[
                    models.FieldCondition(key="turn", range=models.Range(lte=turn)),
                    models.IsEmptyCondition(is_empty=models.PayloadField(key="turn")),
                ]
            )
        )
    return models.Filter(must=must)


async def search_similar_chunks(
    thread_id: str, text_selection: str, turn: int | None = None
) -> list[dict]:
    """Return every chunk of this thread that is relevant to `text_selection`,
    restricted to sources introduced at or before `turn` (None = no cutoff).

    A chunk is relevant if its dense cosine similarity to the claim is at least
    `_MIN_SCORE`. Candidates are ordered by reciprocal-rank fusion of the dense
    and BM25 rankings; `score` is the dense cosine similarity.
    """
    try:
        await asyncio.to_thread(_ensure_ready)
        dense_vec, sparse_vec = await _embed_query(text_selection)
        flt = _filter(thread_id, turn)
        dense_res, sparse_res = await _get_async_client().query_batch_points(
            _COLLECTION,
            requests=[
                models.QueryRequest(
                    query=dense_vec, using="dense", filter=flt, limit=_QUERY_LIMIT, with_payload=True
                ),
                models.QueryRequest(
                    query=models.SparseVector(
                        indices=sparse_vec["indices"], values=sparse_vec["values"]
                    ),
                    using="sparse",
                    filter=flt,
                    limit=_QUERY_LIMIT,
                    with_payload=True,
                ),
            ],
        )
    except Exception as exc:
        logger.warning(f"[vector_sources] search failed for thread {thread_id}: {exc}")
        return []

    dense_points, sparse_points = dense_res.points, sparse_res.points
    print(
        f"[vector_sources] search thread={thread_id} turn_cutoff={turn} "
        f"query={text_selection[:80]!r} -> {len(dense_points)} dense / "
        f"{len(sparse_points)} bm25 candidate(s)"
    )

    fused: dict = collections.defaultdict(float)
    for ranking in (dense_points, sparse_points):
        for rank, p in enumerate(ranking):
            fused[p.id] += 1.0 / (_RRF_K + rank + 1)

    matches = []
    for p in dense_points:  # only chunks with a dense score can pass the gate
        kept = p.score >= _MIN_SCORE
        payload = p.payload or {}
        print(
            f"[vector_sources]   cos={p.score:.4f} fused={fused[p.id]:.4f} turn={payload.get('turn')} "
            f"n={payload.get('n')} kept={kept} text={payload.get('text', '')[:60]!r}"
        )
        if not kept:
            continue
        matches.append(
            {
                "n": payload.get("n"),
                "title": payload.get("title", ""),
                "url": payload.get("url", ""),
                "chunk": payload.get("text", ""),
                "score": round(float(p.score), 4),
                "turn": payload.get("turn"),
                "credibility": payload.get("credibility"),
                "_fused": fused[p.id],
            }
        )
    matches.sort(key=lambda m: m["_fused"], reverse=True)
    for m in matches:
        del m["_fused"]
    return matches


# ── deletion ────────────────────────────────────────────────────────────────


def delete_thread_vectors(thread_id: str) -> None:
    """Remove every indexed chunk for a hard-deleted thread."""
    delete_threads_vectors([thread_id])


def delete_threads_vectors(thread_ids: list[str]) -> None:
    """Remove every indexed chunk for several threads in one request."""
    if not thread_ids:
        return
    _ensure_ready()
    _get_sync_client().delete(
        _COLLECTION,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="thread_id", match=models.MatchAny(any=thread_ids))]
            )
        ),
    )


def list_indexed_thread_ids(older_than_seconds: float = 0) -> list[str]:
    """Distinct thread ids that have chunks in the index. With `older_than_seconds`,
    only threads that have at least one chunk indexed that long ago (a grace period
    so a thread being created right now is never mistaken for an orphan)."""
    _ensure_ready()
    flt = None
    if older_than_seconds:
        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="indexed_at", range=models.Range(lt=time.time() - older_than_seconds)
                )
            ]
        )
    res = _get_sync_client().facet(
        _COLLECTION, key="thread_id", facet_filter=flt, limit=100_000, exact=True
    )
    # Facet keeps listing values whose points were all deleted, with a count of 0.
    return [str(hit.value) for hit in res.hits if hit.count > 0]
