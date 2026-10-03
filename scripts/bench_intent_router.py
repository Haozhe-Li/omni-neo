"""Latency benchmark for core/intent_router.py.

Three questions, each answered from a *fresh process* where that matters (a warm
HTTP/Redis connection pool would otherwise hide most of a cold start):

  1. cold setup        no cached vectors in Redis: embed every anchor, write them back,
                       fit the linear head
  2. setup with Redis  cached vectors present: connect, GET, decode, fit the head
  3. route             one query: one embedding request + the cosine over the anchors

Route latency is measured sequentially (what a lone request sees) and at a few
concurrency levels (what it looks like when turns overlap). The embedding request
and the local cosine are timed separately so it is clear which one the number is.

Numbers are from wherever this runs, so run it where the backend runs: the point is
the cost of the trip to the embedding service and to Redis from there. From a laptop
those trips dominate and the figures are an upper bound.

It never touches the keys the running app reads: it uses its own Redis prefix
(intent_router_bench:v1:) and deletes what it wrote when it finishes. The only shared
resource it loads is the embedding service — about 150 anchor embeddings per cold run
and a few hundred single-query requests, ~10-30s of light traffic in total.

Env (the same as the backend): REDIS_URL, EMBEDDING_SERVICE_URL.

Run from the repo root:
  python -m scripts.bench_intent_router
  python -m scripts.bench_intent_router --repeats 5 --queries 300
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# Before anything imports core.intent_router (children inherit it): benchmark keys
# live under their own prefix, never the app's.
os.environ.setdefault("INTENT_ROUTER_REDIS_PREFIX", "intent_router_bench:v1:")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

perf = time.perf_counter


# Used when evals/scout_routing/cases.yaml is not in the image (or PyYAML is missing):
# a spread of the shapes the router sees, none of them anchors.
_BUILTIN_QUERIES = [
    "what's the weather in Lisbon tomorrow", "北京明天会下雨吗", "Nvidia stock price today", "特斯拉最近股价怎么样",
    "100 dollars in yen", "美元兑人民币汇率", "who are you", "你是哪个公司做的", "what can you do", "你能做什么",
    "what is a vector database", "how do transformers work", "量子计算最新进展", "日本留学签证怎么办理",
    "best laptops for programming", "compare PostgreSQL and MySQL", "谁发明了电话", "react vs svelte performance",
    "translate this into French", "写一首关于春天的诗", "hi", "谢谢", "write a python function to sort a list",
    "帮我润色一下这段话", "tell me a joke", "what is 15 percent of 240", "how are you", "我今天好累",
    "latest news on OpenAI", "怎么在mac上安装docker", "is intermittent fasting effective", "best time to visit Thailand",
    "how do hurricanes form", "为什么日元一直在贬值", "what languages do you support", "你支持哪些语言",
    "Tokyo vs Osaka for a winter trip", "side effects of ibuprofen", "三百五十乘以十二等于几", "good morning",
]


def _queries(n: int) -> list[str]:
    qs, src = _BUILTIN_QUERIES, "built-in sample"
    try:
        import yaml

        path = ROOT / "evals/scout_routing/cases.yaml"
        qs = [c["q"] for c in yaml.safe_load(path.read_text(encoding="utf-8"))["cases"]]
        src = "evals/scout_routing/cases.yaml"
    except Exception:
        pass
    global _QUERY_SOURCE
    _QUERY_SOURCE = src
    qs = list(qs)
    random.Random(0).shuffle(qs)  # a prefix of the ordered file is all one label
    return (qs * (n // len(qs) + 1))[:n]


_QUERY_SOURCE = "?"


def _pcts(xs: list[float]) -> dict[str, float]:
    s = sorted(xs)
    at = lambda q: s[min(len(s) - 1, int(q * len(s)))]  # noqa: E731
    return {"mean": statistics.mean(s), "p50": at(.5), "p90": at(.9), "p99": at(.99), "max": s[-1], "min": s[0]}


# ── child: one fresh process, one setup ─────────────────────────────────────

async def _child(mode: str) -> dict:
    from core.intent_router import IntentRouter

    q = _queries(5)
    t = perf()
    router = await IntentRouter.build(force=(mode == "cold"))
    setup = perf() - t

    t = perf()
    await router.route(q[0])
    first_route = perf() - t

    t = perf()
    await router.route(q[1])
    second_route = perf() - t

    # Same process, connections now warm: isolates the Redis GET + decode from
    # the cost of opening the connection.
    t = perf()
    await IntentRouter.build(force=False)
    rebuild_from_cache = perf() - t
    return {"setup": setup, "first_route": first_route, "second_route": second_route,
            "rebuild_from_cache": rebuild_from_cache, "fit": router.fit_seconds}


def _spawn(mode: str) -> dict:
    out = subprocess.run(
        [sys.executable, "-m", "scripts.bench_intent_router", "--child", mode],
        cwd=ROOT, capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        raise RuntimeError(f"child {mode} failed:\n{out.stderr[-1500:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


# ── route latency, in this process ──────────────────────────────────────────

async def _route_bench(n: int, levels: list[int]) -> dict:
    from core.intent_router import IntentRouter

    router = await IntentRouter.build()
    qs = _queries(n)
    for q in qs[:5]:  # warm connections, discard
        await router.route(q)

    total, embed, compute = [], [], []
    decided: dict[str, int] = {}
    for q in qs:
        t0 = perf()
        v = (await router.embed_queries([q]))[0]
        t1 = perf()
        res = router.route_vector(v)
        t2 = perf()
        decided[res.label or "unknown"] = decided.get(res.label or "unknown", 0) + 1
        embed.append(t1 - t0)
        compute.append(t2 - t1)
        total.append(t2 - t0)

    conc = {}
    for c in levels:
        sem = asyncio.Semaphore(c)
        lat: list[float] = []

        async def one(q: str) -> None:
            async with sem:
                t = perf()
                await router.route(q)
                lat.append(perf() - t)

        t = perf()
        await asyncio.gather(*(one(q) for q in qs[: max(64, c * 8)]))
        conc[c] = {**_pcts(lat), "wall": perf() - t, "n": len(lat)}
    return {"n": n, "total": _pcts(total), "embed": _pcts(embed), "compute": _pcts(compute), "conc": conc,
            "labels": decided}


# ── report ──────────────────────────────────────────────────────────────────

def ms(x: float) -> str:
    return f"{x * 1000:8.1f}"


def _row(name: str, vals: list[float]) -> str:
    p = _pcts(vals)
    return f"  {name:<44}{ms(p['mean'])}{ms(p['min'])}{ms(p['max'])}   (n={len(vals)})"


def _host(env: str) -> str:
    from urllib.parse import urlparse

    v = os.environ.get(env)
    return (urlparse(v).hostname or "set") if v else "NOT SET"


def _cleanup_bench_keys() -> int:
    """Delete every key this benchmark wrote (its own prefix only).

    The sync client on purpose: the shared async one is bound to the event loop of
    whichever asyncio.run created it first, and that loop is closed by now.
    """
    from core import intent_router as ir
    from core.utils.redis_client import get_redis

    assert ir._KEY_PREFIX.startswith("intent_router_bench:"), "refusing to clean a non-benchmark prefix"
    r = get_redis()
    keys = set(r.smembers(ir._REGISTRY))
    keys.add(ir._REGISTRY)
    r.delete(*keys)
    return len(keys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=["cold", "redis"], help=argparse.SUPPRESS)
    ap.add_argument("--repeats", type=int, default=3, help="fresh-process repetitions per setup mode")
    ap.add_argument("--queries", type=int, default=211, help="sequential route() samples")
    ap.add_argument("--levels", default="4,16", help="concurrency levels for the overlap test")
    a = ap.parse_args()

    if a.child:
        print(json.dumps(asyncio.run(_child(a.child))))
        return

    from core.intent_examples import INTENT_EXAMPLES

    n_anchors = sum(len(v) for v in INTENT_EXAMPLES.values())
    print(f"intent router bench: {n_anchors} anchors, {len(INTENT_EXAMPLES)} labels", flush=True)
    print(f"embedding service host: {_host('EMBEDDING_SERVICE_URL')}   redis host: {_host('REDIS_URL')}   "
          f"redis prefix: {os.environ['INTENT_ROUTER_REDIS_PREFIX']}", flush=True)
    if "NOT SET" in (_host("EMBEDDING_SERVICE_URL"), _host("REDIS_URL")):
        sys.exit("EMBEDDING_SERVICE_URL and REDIS_URL must both be set")

    cold, warm = [], []
    try:
        # Make sure the "redis" runs really find a cache: one priming cold run first.
        print("priming cache...", flush=True)
        _spawn("cold")
        for i in range(a.repeats):
            print(f"fresh process {i + 1}/{a.repeats}: cold, then with Redis", flush=True)
            cold.append(_spawn("cold"))
            warm.append(_spawn("redis"))

        print("route latency...", flush=True)
        rb = asyncio.run(_route_bench(a.queries, [int(x) for x in a.levels.split(",")]))
    finally:
        try:
            print(f"cleaned up {_cleanup_bench_keys()} benchmark Redis keys", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not delete benchmark keys under {os.environ['INTENT_ROUTER_REDIS_PREFIX']}: {exc!r}")

    print("\n" + "=" * 78)
    print(f"  {'(milliseconds)':<44}{'mean':>8}{'min':>8}{'max':>8}")
    print("\n  Setup (fresh process each)")
    print(_row("cold: no cache, embed all anchors + write Redis", [c["setup"] for c in cold]))
    print(_row("with Redis: connect + GET + decode", [w["setup"] for w in warm]))
    print(_row("  same, 2nd build in-process (conn warm)", [w["rebuild_from_cache"] for w in warm]))

    print(_row("  of which: fit the linear head (CPU)", [w["fit"] for w in warm]))

    print("\n  First requests in a fresh process")
    print(_row("1st route() after cold setup", [c["first_route"] for c in cold]))
    print(_row("1st route() after Redis setup", [w["first_route"] for w in warm]))
    print(_row("2nd route() after Redis setup", [w["second_route"] for w in warm]))

    t, e, c = rb["total"], rb["embed"], rb["compute"]
    print(f"\n  route(), sequential, warm, n={rb['n']}      {'mean':>8}{'p50':>8}{'p90':>8}{'p99':>8}{'max':>8}")
    for name, p in (("total", t), ("  embedding request", e), ("  head + nearest-anchor cosine (local)", c)):
        print(f"  {name:<36}" + "".join(ms(p[k]) for k in ("mean", "p50", "p90", "p99", "max")))

    print(f"\n  route() with overlapping requests    {'mean':>8}{'p50':>8}{'p90':>8}{'p99':>8}{'max':>8}   wall/req")
    for lvl, p in rb["conc"].items():
        print(f"  {lvl:>2} in flight (n={p['n']:<4})              "
              + "".join(ms(p[k]) for k in ("mean", "p50", "p90", "p99", "max"))
              + f"   {p['wall'] / p['n'] * 1000:6.1f}ms")
    lab = rb["labels"]
    tot = sum(lab.values())
    unknown = lab.get("unknown", 0)
    print(f"\n  Router verdicts on the {tot} queries above ({_QUERY_SOURCE}), shipped thresholds:")
    print("  " + "  ".join(f"{k} {v / tot:.0%}" for k, v in sorted(lab.items(), key=lambda x: -x[1])))
    print(f"  -> decided by the router {1 - unknown / tot:.0%}, unknown (goes to the LLM scout) {unknown / tot:.0%}")
    print("=" * 78)


if __name__ == "__main__":
    main()
