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

Output. Progress is one line per phase; everything else is collected and printed as a
single report at the end (tables for the three parts, a short summary, and the cleanup
status). The backend's own per-request log lines are muted during the run.

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
import logging
import os
import random
import statistics
import sys
import time
import uuid
import warnings

warnings.filterwarnings("ignore")  # library deprecation noise would bury the report

RUN_ID = uuid.uuid4().hex[:8]
SCRATCH = f"omni_bench_{RUN_ID}"
PROD_DEFAULT = "omni_chunks"
# Must be set before vector_sources is imported: it reads the name at import time.
os.environ["VECTOR_COLLECTION"] = SCRATCH

from qdrant_client import AsyncQdrantClient, models  # noqa: E402

from core.utils import redis_sources, vector_sources as vs  # noqa: E402
from core.utils.redis_client import get_redis  # noqa: E402

assert vs._COLLECTION == SCRATCH and SCRATCH != PROD_DEFAULT, "refusing to run against a real collection"

# vector_sources print()s a line per indexed source and per search candidate; mute that
# (a module-level name shadows the builtin inside that module only).
vs.print = lambda *a, **k: None


class _WarnCounter(logging.Handler):
    """Collects warnings instead of letting them interleave with the progress lines."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.count = 0
        self.first: list[str] = []

    def emit(self, record):
        self.count += 1
        msg = record.getMessage()[:160]
        if len(self.first) < 3 and msg not in self.first:
            self.first.append(msg)


WARNINGS = _WarnCounter()
logging.getLogger().addHandler(WARNINGS)


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def stats(values: list[float]) -> tuple[float, float, float, float]:
    """(mean, p50, p95, max) in ms from seconds."""
    return (
        statistics.mean(values) * 1000,
        pct(values, .5) * 1000,
        pct(values, .95) * 1000,
        max(values) * 1000,
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


def progress(msg: str) -> None:
    print(msg, flush=True)


async def amain(args, R: dict) -> None:
    """Runs the three parts and stores the measurements in R; prints only progress."""
    sync = vs._get_sync_client()

    # ── pre-flight ─────────────────────────────────────────────────────────
    health = vs._get_embed_sync().get(f"{vs._embed_url()}/health").json()
    if vs.DENSE_MODEL not in health.get("text_models", {}):
        sys.exit(f"embedding service does not serve {vs.DENSE_MODEL}")
    R["service"] = (
        f"{vs.DENSE_MODEL.split('/')[-1]} + BM25, queue {health.get('queue_depth')}/{health.get('queue_max')}, "
        f"rss {health.get('rss_mb')}MB"
    )

    # ── seed ───────────────────────────────────────────────────────────────
    progress("seeding scratch collection from real sources ...")
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
    R["seed"] = (len(seeded), len(srcs), n_points, n_points / dt, dt / len(srcs) * 1000)
    claims = make_claims(seeded, args.claims, args.seed)
    texts = [c["claim"] for c in claims]
    reps = args.repeat

    # ── 1. embedding service ───────────────────────────────────────────────
    progress("[1/3] embedding service ...")
    R["embed"] = []
    dense_p = lambda t: {"texts": [t], "model": vs.DENSE_MODEL}  # noqa: E731
    sparse_p = lambda t: {"texts": [t], "is_query": True}  # noqa: E731
    for label, fn in (
        ("query, dense", lambda t: vs._post_async("/embed/dense/text", dense_p(t))),
        ("query, BM25", lambda t: vs._post_async("/embed/sparse", sparse_p(t))),
        ("query, dense + BM25 together (one search)", lambda t: vs._embed_query(t)),
    ):
        lat = []
        for i in range(reps):
            s = time.perf_counter()
            await fn(texts[i % len(texts)])
            lat.append(time.perf_counter() - s)
        R["embed"].append((label, stats(lat)))
    chunk_texts = [p.payload["text"] for p in sync.scroll(SCRATCH, limit=32, with_payload=["text"])[0]]
    lat = []
    for _ in range(max(5, reps // 3)):
        s = time.perf_counter()
        await asyncio.to_thread(vs._embed_docs_sync, chunk_texts)
        lat.append(time.perf_counter() - s)
    R["embed"].append((f"document batch, {len(chunk_texts)} chunks (indexing)", stats(lat)))

    # ── 2. Qdrant ──────────────────────────────────────────────────────────
    progress("[2/3] Qdrant ...")
    R["qdrant"] = []
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
    R["qdrant"].append(("network round trip (baseline)", stats(lat), None))
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
        R["qdrant"].append((f"hybrid query, concurrency {level}", stats(lat), total / (time.perf_counter() - w0)))

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
    R["qdrant"].append((f"upsert one source ({len(pts)} chunks)", stats(up), None))
    R["qdrant"].append(("delete one thread", stats(dele), None))

    # ── 3. check_source end to end ─────────────────────────────────────────
    if args.no_e2e:
        progress("[3/3] skipped (--no-e2e)")
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

    R["e2e"] = []
    for level in args.levels:
        progress(f"[3/3] check_source end to end, concurrency {level} ({len(claims)} calls) ...")
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
        mean = lambda k: statistics.mean(stages[k]) * 1000 if stages[k] else 0.0  # noqa: E731
        total_mean = statistics.mean(lat) * 1000
        R["e2e"].append({
            "level": level, "stats": stats(lat), "rate": len(claims) / wall, "n": len(claims),
            "embed": mean("embed"), "qdrant": mean("qdrant"), "rerank": mean("rerank"),
            "other": max(0.0, total_mean - mean("embed") - mean("qdrant") - mean("rerank")),
            "found": found,
        })


def cleanup(name: str) -> str:
    client = vs._get_sync_client()
    if name == PROD_DEFAULT or not name.startswith("omni_bench_"):
        return f"refused to delete {name!r}"
    msg = f"{name} did not exist"
    if client.collection_exists(name):
        n = client.count(name, exact=True).count
        client.delete_collection(name)
        msg = f"deleted scratch collection {name} ({n} points)"
    gone = not client.collection_exists(name)
    return msg + (", verified gone" if gone else ", STILL PRESENT - delete it by hand")


def report(R: dict, args, cleanup_msg: str) -> None:
    W = 74
    f1 = lambda v: f"{v:7.0f}"  # noqa: E731
    line = lambda label, st, extra="": print(  # noqa: E731
        f"  {label:<46}{f1(st[1])}{f1(st[2])}{f1(st[3])}  {extra}"
    )
    print("\n" + "=" * W)
    print("VECTOR SEARCH BENCHMARK".center(W))
    print("=" * W)
    if "seed" in R:
        th, src, ch, cps, mps = R["seed"]
        print(f"data      {th} real threads, {src} sources, {ch} chunks (scratch collection)")
        print(f"indexing  {cps:.0f} chunks/s, {mps:.0f} ms per source (4 workers)")
        print(f"embedding {R['service']}")
        print(f"qdrant    {os.environ.get('QDRANT_URL', '')}")

    hdr = f"  {'':<46}{'p50':>7}{'p95':>7}{'max':>7}  (ms)"
    if "embed" in R:
        print("\n1. EMBEDDING SERVICE\n" + hdr)
        for label, st in R["embed"]:
            line(label, st)
    if "qdrant" in R:
        print("\n2. QDRANT\n" + hdr)
        for label, st, rate in R["qdrant"]:
            line(label, st, f"{rate:.0f}/s" if rate else "")

    e2e = R.get("e2e")
    if e2e:
        print("\n3. /check_source END TO END (claim in, matches out; includes LLM rerank)")
        print(f"  {'':<14}{'p50':>7}{'p95':>7}{'max':>7}   {'embed':>6}{'qdrant':>7}{'rerank':>7}{'other':>6}   {'rate':>6}  found-source")
        for r in e2e:
            st = r["stats"]
            print(
                f"  concurrency {r['level']:<2}{f1(st[1])}{f1(st[2])}{f1(st[3])}   {r['embed']:6.0f}{r['qdrant']:7.0f}"
                f"{r['rerank']:7.0f}{r['other']:6.0f}   {r['rate']:4.1f}/s  {r['found']}/{r['n']}"
            )
        print("  (ms; embed/qdrant/rerank = mean time spent in that stage per call, including queueing)")

    print("\nSUMMARY")
    if "embed" in R:
        one = next(st for lbl, st in R["embed"] if "one search" in lbl)
        print(f"  - embedding one search query: p50 {one[1]:.0f} ms, p95 {one[2]:.0f} ms")
    if "qdrant" in R:
        base = R["qdrant"][0][1]
        hy = [(lbl, st, rate) for lbl, st, rate in R["qdrant"] if lbl.startswith("hybrid")]
        best = max(hy, key=lambda x: x[2])
        print(f"  - Qdrant hybrid query: p50 {hy[0][1][1]:.0f} ms at concurrency 1 (round trip alone {base[1]:.0f} ms); "
              f"peak {best[2]:.0f} queries/s at {best[0].split()[-1]} concurrent")
    if e2e:
        r = e2e[0]
        mid = r["embed"] + r["qdrant"]
        tot = r["stats"][0]
        print(f"  - check_source at concurrency {r['level']}: p50 {r['stats'][1]:.0f} ms, mean {tot:.0f} ms = "
              f"search {mid:.0f} (embed {r['embed']:.0f} + qdrant {r['qdrant']:.0f}) + LLM rerank {r['rerank']:.0f} + other {r['other']:.0f}")
        print(f"  - the vector search part is {mid / tot * 100:.0f}% of the end-to-end time; the LLM rerank is {r['rerank'] / tot * 100:.0f}%")
        print(f"  - found the source in {sum(x['found'] for x in e2e)}/{sum(x['n'] for x in e2e)} calls (self-check only)")
    if WARNINGS.count:
        print(f"  - {WARNINGS.count} warning(s) were logged during the run, e.g.:")
        for m in WARNINGS.first:
            print(f"      {m}")
    print(f"\ncleanup: {cleanup_msg}")
    print("=" * W)


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
                print(cleanup(c.name))
        return

    R: dict = {}
    try:
        asyncio.run(amain(args, R))
    finally:
        # Always remove the scratch data, and still show whatever was measured.
        cleanup_msg = cleanup(SCRATCH)
        report(R, args, cleanup_msg)


if __name__ == "__main__":
    main()
