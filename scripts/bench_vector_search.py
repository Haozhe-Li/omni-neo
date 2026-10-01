"""Latency benchmark for the /check_source vector path. Meant to be run inside the
private network, so the numbers exclude public-internet latency.

It measures, in order:
  1. the embedding service on its own (query dense, query BM25, both together, a
     32-chunk document batch);
  2. Qdrant on its own (the exact hybrid query check_source sends, at several
     concurrencies, plus upsert and delete);
  3. check_source end to end — claim in, matches out — where embedding, Qdrant and the
     LLM excerpt-rerank are the middle of the pipeline, with a per-stage breakdown.

Data. It copies a sample of REAL sources from Redis (omni:citation:*) into a throwaway
Qdrant collection named omni_bench_<run id> under synthetic thread ids, so chunk sizes
and languages are realistic. The production collection is never touched. The scratch
collection is deleted when the run ends, also on error or Ctrl-C (use --cleanup-stale
to remove one left behind by a killed run).

Claims are sentences cut from the indexed chunks (no LLM involved in making them), so
this measures latency; the "found source" figure is only a sanity check, not a quality
metric.

Env (the same as the backend): REDIS_URL, QDRANT_URL, QDRANT_API_KEY,
EMBEDDING_SERVICE_URL, GROQ_API_KEY (the rerank step calls the LLM; skip it with
--no-e2e).

Run from the repo root:
  python -m scripts.bench_vector_search
  python -m scripts.bench_vector_search --claims 40 --levels 1,4,8,16
  python -m scripts.bench_vector_search --no-e2e          # no LLM calls
  python -m scripts.bench_vector_search --cleanup-stale   # remove leftover scratch collections
"""

import argparse
import asyncio
import os
import random
import statistics
import sys
import time
import uuid

RUN_ID = uuid.uuid4().hex[:8]
SCRATCH = f"omni_bench_{RUN_ID}"
PROD_DEFAULT = "omni_chunks"
# Must be set before vector_sources is imported: it reads the name at import time.
os.environ["VECTOR_COLLECTION"] = SCRATCH

from qdrant_client import AsyncQdrantClient, models  # noqa: E402

from core.utils import redis_sources, vector_sources as vs  # noqa: E402
from core.utils.redis_client import get_redis  # noqa: E402

assert vs._COLLECTION == SCRATCH and SCRATCH != PROD_DEFAULT, "refusing to run against a real collection"


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def row(label: str, values: list[float], unit: float = 1000.0) -> None:
    """Latencies in seconds -> ms."""
    print(
        f"  {label:<44} n={len(values):<4} mean={statistics.mean(values) * unit:7.1f}  "
        f"p50={pct(values, .5) * unit:7.1f}  p95={pct(values, .95) * unit:7.1f}  "
        f"max={max(values) * unit:7.1f}  ms"
    )


def thread_has_enough(items: list[dict]) -> bool:
    return len([i for i in items if len(i.get("content") or "") > 300]) >= 4


def load_sample(n_threads: int, seed: int) -> list[tuple[str, list[dict]]]:
    r = get_redis()
    ids = [k[len("omni:citation:"):] for k in r.scan_iter("omni:citation:*", count=500) if not k.endswith(":index")]
    random.Random(seed).shuffle(ids)
    out = []
    for tid in ids:
        items = [
            i for i in redis_sources.load_citations(tid)
            if (i.get("credibility") or {}).get("label") != "junk" and (i.get("content") or "").strip()
        ]
        if thread_has_enough(items):
            out.append((tid, items))
        if len(out) == n_threads:
            break
    return out


def make_claims(seeded: list[tuple[str, list[dict]]], count: int, seed: int) -> list[dict]:
    """A sentence-sized excerpt from a random chunk of a random source."""
    rnd = random.Random(seed)
    pool = []
    for bench_tid, items in seeded:
        for rec in items:
            chunks = [c for c in vs._splitter.split_text(rec["content"].strip()[: vs._MAX_INDEX_CHARS]) if len(c) > 150]
            if chunks:
                pool.append((bench_tid, rec["n"], chunks))
    rnd.shuffle(pool)
    claims = []
    for tid, n, chunks in pool[:count]:
        c = rnd.choice(chunks)
        claims.append({"thread": tid, "n": n, "claim": c[: min(len(c), 180)].strip()})
    return claims


async def amain(args) -> None:
    sync = vs._get_sync_client()

    # ── pre-flight ─────────────────────────────────────────────────────────
    health = vs._get_embed_sync().get(f"{vs._embed_url()}/health").json()
    print(f"embedding service: {health.get('dense_model')} text_models={list(health.get('text_models', {}))} "
          f"queue={health.get('queue_depth')}/{health.get('queue_max')} rss={health.get('rss_mb')}MB")
    if vs.DENSE_MODEL not in health.get("text_models", {}):
        sys.exit(f"embedding service does not serve {vs.DENSE_MODEL}")
    print(f"qdrant: {os.environ['QDRANT_URL']}  scratch collection: {SCRATCH}")

    # ── seed ───────────────────────────────────────────────────────────────
    sample = load_sample(args.threads, args.seed)
    if not sample:
        sys.exit("no threads with enough sources in Redis (omni:citation:*)")
    seeded = [(f"bench-{RUN_ID}-{i}", items) for i, (_, items) in enumerate(sample)]
    srcs = [(t, r) for t, items in seeded for r in items]
    vs._ensure_ready()
    t0 = time.perf_counter()
    for f in [vs._executor.submit(vs._index_source_sync, t, r) for t, r in srcs]:
        f.result()
    dt = time.perf_counter() - t0
    n_points = sync.count(SCRATCH, exact=True).count
    print(f"\nseeded {len(seeded)} threads, {len(srcs)} sources, {n_points} chunks "
          f"in {dt:.1f}s ({n_points / dt:.0f} chunks/s, {dt / len(srcs) * 1000:.0f} ms/source, "
          f"4 indexing workers)")
    claims = make_claims(seeded, args.claims, args.seed)
    texts = [c["claim"] for c in claims]
    reps = args.repeat

    # ── 1. embedding service ───────────────────────────────────────────────
    print(f"\n[1] embedding service alone ({reps} sequential requests each)")
    dense_p = lambda t: {"texts": [t], "model": vs.DENSE_MODEL}  # noqa: E731
    sparse_p = lambda t: {"texts": [t], "is_query": True}  # noqa: E731
    for label, fn in (
        ("query dense (1 text)", lambda t: vs._post_async("/embed/dense/text", dense_p(t))),
        ("query BM25 (1 text)", lambda t: vs._post_async("/embed/sparse", sparse_p(t))),
        ("query dense + BM25 in parallel (= one search)", lambda t: vs._embed_query(t)),
    ):
        lat = []
        for i in range(reps):
            s = time.perf_counter()
            await fn(texts[i % len(texts)])
            lat.append(time.perf_counter() - s)
        row(label, lat)
    chunk_texts = [
        p.payload["text"] for p in sync.scroll(SCRATCH, limit=32, with_payload=["text"])[0]
    ]
    lat = []
    for _ in range(max(5, reps // 3)):
        s = time.perf_counter()
        await asyncio.to_thread(vs._embed_docs_sync, chunk_texts)
        lat.append(time.perf_counter() - s)
    row(f"document batch ({len(chunk_texts)} chunks, dense + BM25)", lat)

    # ── 2. Qdrant ──────────────────────────────────────────────────────────
    print(f"\n[2] Qdrant alone (query vectors embedded beforehand)")
    qvecs = [await vs._embed_query(t) for t in texts]
    aclient = vs._get_async_client()

    async def hybrid(i: int) -> None:
        dv, sv = qvecs[i % len(qvecs)]
        flt = vs._filter(claims[i % len(claims)]["thread"], None)
        await aclient.query_batch_points(
            SCRATCH,
            requests=[
                models.QueryRequest(query=dv, using="dense", filter=flt, limit=vs._QUERY_LIMIT, with_payload=True),
                models.QueryRequest(
                    query=models.SparseVector(indices=sv["indices"], values=sv["values"]),
                    using="sparse", filter=flt, limit=vs._QUERY_LIMIT, with_payload=True,
                ),
            ],
        )

    lat = []
    for _ in range(reps):
        s = time.perf_counter()
        await aclient.collection_exists(SCRATCH)
        lat.append(time.perf_counter() - s)
    row("round trip (collection_exists)", lat)
    for level in args.levels:
        total = max(reps, level * 6)
        lat, sem = [], asyncio.Semaphore(level)

        async def one(i: int) -> None:
            async with sem:
                s = time.perf_counter()
                await hybrid(i)
                lat.append(time.perf_counter() - s)

        w0 = time.perf_counter()
        await asyncio.gather(*[one(i) for i in range(total)])
        wall = time.perf_counter() - w0
        row(f"hybrid query, concurrency {level} ({total / wall:.0f}/s)", lat)

    # upsert / delete of one realistic source (5 chunks, vectors reused)
    pts = sync.scroll(SCRATCH, limit=5, with_vectors=True, with_payload=True)[0]
    up, dele = [], []
    for k in range(10):
        tid = f"bench-{RUN_ID}-tmp{k}"
        batch = [
            models.PointStruct(id=str(uuid.uuid4()), vector=p.vector, payload={**p.payload, "thread_id": tid})
            for p in pts
        ]
        s = time.perf_counter()
        await asyncio.to_thread(sync.upsert, SCRATCH, points=batch)
        up.append(time.perf_counter() - s)
        s = time.perf_counter()
        await asyncio.to_thread(vs.delete_threads_vectors, [tid])
        dele.append(time.perf_counter() - s)
    row(f"upsert {len(pts)} chunks", up)
    row("delete one thread (by filter)", dele)

    # ── 3. check_source end to end ─────────────────────────────────────────
    if args.no_e2e:
        print("\n[3] skipped (--no-e2e)")
        return
    from core import check_source as cs
    from core.utils import source_rerank

    # Bypass the 10-day Redis cache so every call is real and nothing is left in Redis.
    raw = getattr(cs._check_source_matches_cached, "__wrapped__", None)
    if raw is None:
        sys.exit("cannot bypass check_source's cache (no __wrapped__); aborting rather than polluting Redis")

    stages: dict[str, list[float]] = {"embed": [], "qdrant": [], "rerank": []}

    def timed(name, fn):
        async def wrapper(*a, **k):
            s = time.perf_counter()
            try:
                return await fn(*a, **k)
            finally:
                stages[name].append(time.perf_counter() - s)
        return wrapper

    vs._embed_query = timed("embed", vs._embed_query)
    AsyncQdrantClient.query_batch_points = timed("qdrant", AsyncQdrantClient.query_batch_points)
    source_rerank.rerank_candidates = timed("rerank", source_rerank.rerank_candidates)

    print(f"\n[3] check_source end to end ({len(claims)} claims per level; includes the LLM rerank)")
    print(f"  {'':<14}{'p50':>9}{'p95':>9}{'max':>9}   mean per stage (ms): embed / qdrant / rerank   found-source")
    for level in args.levels:
        for v in stages.values():
            v.clear()
        lat, found, nonempty, sem = [], 0, 0, asyncio.Semaphore(level)

        async def call(c: dict) -> None:
            nonlocal found, nonempty
            async with sem:
                s = time.perf_counter()
                matches = await raw(c["thread"], c["claim"], None)
                lat.append(time.perf_counter() - s)
            nonempty += bool(matches)
            found += any(m["n"] == c["n"] for m in matches)

        w0 = time.perf_counter()
        await asyncio.gather(*[call(c) for c in claims])
        wall = time.perf_counter() - w0
        m = lambda k: statistics.mean(stages[k]) * 1000 if stages[k] else float("nan")  # noqa: E731
        print(f"  concurrency {level:<2}{pct(lat, .5) * 1000:8.0f}ms{pct(lat, .95) * 1000:7.0f}ms{max(lat) * 1000:7.0f}ms"
              f"   {m('embed'):7.0f} / {m('qdrant'):6.0f} / {m('rerank'):6.0f}"
              f"          {found}/{len(claims)}  ({len(claims) / wall:.1f}/s)")
    print("  (the stage means are per call of that stage; at concurrency > 1 they include queueing)")


def cleanup(name: str) -> None:
    client = vs._get_sync_client()
    if name == PROD_DEFAULT or not name.startswith("omni_bench_"):
        print(f"refusing to delete {name!r}")
        return
    if client.collection_exists(name):
        n = client.count(name, exact=True).count
        client.delete_collection(name)
        print(f"deleted scratch collection {name} ({n} points)")
    gone = not client.collection_exists(name)
    print(f"cleanup verified: {'collection is gone' if gone else 'STILL PRESENT, delete it by hand'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=10, help="real threads to copy into the scratch collection")
    ap.add_argument("--claims", type=int, default=24, help="claims for the end-to-end test (each is one LLM rerank call)")
    ap.add_argument("--repeat", type=int, default=30, help="sequential samples for the single-request timings")
    ap.add_argument("--levels", default="1,4,8", help="concurrency levels, comma separated")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-e2e", action="store_true", help="skip section 3 (no LLM calls)")
    ap.add_argument("--cleanup-stale", action="store_true", help="delete leftover omni_bench_* collections and exit")
    args = ap.parse_args()
    args.levels = [int(x) for x in args.levels.split(",")]

    if args.cleanup_stale:
        client = vs._get_sync_client()
        for c in client.get_collections().collections:
            if c.name.startswith("omni_bench_"):
                cleanup(c.name)
        return

    try:
        asyncio.run(amain(args))
    finally:
        print()
        cleanup(SCRATCH)


if __name__ == "__main__":
    main()
