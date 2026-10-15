"""Scout routing eval — baseline for the keyword shortcut, optionally the LLM scout.

    venv/bin/python3.12 evals/scout_routing/run.py            # keyword stage only, offline
    venv/bin/python3.12 evals/scout_routing/run.py --llm      # + the production scout model
    venv/bin/python3.12 evals/scout_routing/run.py --router   # + the Decisions-API intent router
                                                              #   (needs OPENAI_API_KEY)

What is scored, and why it is two different things:

* **Keyword stage** (`shortcut_decision`) can only answer `web_search`,
  `about_omni`, or *defer* (None = hand the query to the LLM). Deferring is never
  a wrong answer, only a slow one, so it is scored separately from being wrong:

    hit            decided, and it matches gold
    defer_ok       deferred, and gold is something the keyword stage never decides
                   (weather/stock/currency/direct_response) — the intended outcome
    miss           deferred, but gold is web_search/about_omni — correct but pays the
                   LLM round trip the shortcut exists to avoid
    wrong_widget   decided web_search/about_omni on a weather/stock/currency query —
                   the user silently loses the live card (the worst outcome)
    wrong_identity decided about_omni for a non-Omni query, or web_search for an
                   Omni question — injects the wrong block
    wrong_fp       decided web_search on a direct_response query — one wasted search
                   plus a block of results the agent has to ignore

* **LLM scout** (`--llm`) is scored by plain accuracy on the same gold, plus
  argument extraction for weather/stock/currency, and wall-clock latency. The
  production pipeline is keyword-then-LLM, so that composite is reported too.

LangSmith tracing is forced off: `classify()` logs to the production
`context-enrichment` project, which is where real scout decisions are mined from,
and eval traffic must not land in it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import yaml  # noqa: E402

HERE = Path(__file__).parent
INTENTS = ["web_search", "weather", "stock", "currency", "about_omni", "direct_response", "skill:web-research"]
WIDGET_INTENTS = {"weather", "stock", "currency"}
# What the keyword stage is able to say. Everything else is, by design, a defer.
KEYWORD_DECIDES = {"web_search", "about_omni"}

ACTION_TO_INTENT = {
    "web_search": "web_search",
    "weather_current": "weather",
    "weather_forecast": "weather",
    "stock": "stock",
    "currency": "currency",
    "about_omni": "about_omni",
    "direct_response": "direct_response",
}


def load_cases() -> list[dict]:
    from core.context_enrichment import word_count, _WORD_LIMIT, _SHORTCUT_WORD_LIMIT

    raw = yaml.safe_load((HERE / "cases.yaml").read_text(encoding="utf-8"))["cases"]
    cases = []
    for i, c in enumerate(raw):
        assert c["gold"] in INTENTS, f"case {i}: bad gold {c['gold']!r}"
        wc = word_count(c["q"])
        tags = set(c.get("tags") or [])
        if wc >= _SHORTCUT_WORD_LIMIT:
            tags.add("long")
        assert wc <= _WORD_LIMIT, f"case {i} is over the scout's own length gate: {c['q'][:40]!r}"
        cases.append({**c, "id": i, "tags": sorted(tags), "words": wc})
    return cases


# ── keyword stage ───────────────────────────────────────────────────────────

def run_keyword(cases: list[dict]) -> list[dict]:
    from core.context_enrichment import shortcut_decision

    out = []
    for c in cases:
        d = shortcut_decision(c["q"])
        pred = ACTION_TO_INTENT[d.action] if d else None
        out.append({"id": c["id"], "pred": pred, "outcome": keyword_outcome(c["gold"], pred)})
    return out


def keyword_outcome(gold: str, pred: str | None) -> str:
    if pred is None:
        return "miss" if gold in KEYWORD_DECIDES else "defer_ok"
    if pred == gold:
        return "hit"
    if gold in WIDGET_INTENTS:
        return "wrong_widget"
    if pred == "about_omni" or gold == "about_omni":
        return "wrong_identity"
    return "wrong_fp"  # pred web_search, gold direct_response


KW_OUTCOMES = ["hit", "defer_ok", "miss", "wrong_widget", "wrong_identity", "wrong_fp"]
KW_WRONG = {"wrong_widget", "wrong_identity", "wrong_fp"}


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:5.1f}%" if d else "  n/a"


def keyword_report(cases, results, label_filter=None, title="") -> None:
    pairs = [(c, r) for c, r in zip(cases, results) if label_filter is None or label_filter(c)]
    n = len(pairs)
    oc = Counter(r["outcome"] for _, r in pairs)
    decided = oc["hit"] + sum(oc[k] for k in KW_WRONG)
    should_decide = sum(1 for c, _ in pairs if c["gold"] in KEYWORD_DECIDES)

    print(f"\n── Keyword stage{title} — {n} cases ──")
    print(f"  decided {decided}  deferred {n - decided}")
    print(f"  precision (decided & correct / decided)      {pct(oc['hit'], decided)}   ({oc['hit']}/{decided})")
    print(f"  recall    (decided & correct / should-decide) {pct(oc['hit'], should_decide)}   ({oc['hit']}/{should_decide})")
    print(f"  wrong answers                                 {sum(oc[k] for k in KW_WRONG)}   "
          f"(widget {oc['wrong_widget']}, identity {oc['wrong_identity']}, false-search {oc['wrong_fp']})")
    print(f"  deferred as intended {oc['defer_ok']}   missed (correct but slow) {oc['miss']}")

    print(f"\n  {'gold':<16}{'n':>4}" + "".join(f"{k:>16}" for k in KW_OUTCOMES))
    by_gold: dict[str, Counter] = defaultdict(Counter)
    for c, r in pairs:
        by_gold[c["gold"]][r["outcome"]] += 1
    for g in INTENTS:
        row = by_gold[g]
        if not row:
            continue
        print(f"  {g:<16}{sum(row.values()):>4}" + "".join(f"{row[k]:>16}" for k in KW_OUTCOMES))


def keyword_details(cases, results) -> None:
    print("\n── Keyword stage: wrong answers ──")
    for c, r in zip(cases, results):
        if r["outcome"] in KW_WRONG:
            print(f"  [{r['outcome']:<14}] gold={c['gold']:<15} pred={r['pred']:<10} {c['q'][:70]!r}")
    print("\n── Keyword stage: misses (gold web_search/about_omni but deferred) ──")
    for c, r in zip(cases, results):
        if r["outcome"] == "miss":
            tag = " long" if "long" in c["tags"] else ""
            print(f"  gold={c['gold']:<11}{tag:<5} {c['q'][:80]!r}")


# ── LLM scout ───────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return (s or "").strip().lower()


def args_ok(case: dict, d) -> bool | None:
    """Whether the extracted arguments match gold. None when gold has no args."""
    want = case.get("args")
    if not want:
        return None
    g = case["gold"]
    if g == "weather":
        got, exp = _norm(d.location), _norm(want["location"])
        return bool(got) and (exp in got or got in exp)
    if g == "stock":
        return _norm(d.ticker) == _norm(want["ticker"])
    if g == "currency":
        return _norm(d.base_currency) == _norm(want["base"]) and _norm(d.target_currency) == _norm(want["target"])
    return None


async def run_llm(cases: list[dict], concurrency: int, timeout: float) -> list[dict]:
    from langsmith import tracing_context
    from core.context_enrichment import _llm, build_scout_messages

    sem = asyncio.Semaphore(concurrency)

    async def one(c: dict) -> dict:
        loc = (c.get("ctx") or {}).get("location")
        # Same prompt/messages and model as `classify`, but without its
        # production tracing project (see module docstring).
        messages = build_scout_messages(c["q"], user_location=loc)
        async with sem:
            t0 = time.monotonic()
            try:
                with tracing_context(enabled=False):
                    d = await asyncio.wait_for(_llm.ainvoke(messages), timeout=timeout)
                err = None
            except Exception as e:  # noqa: BLE001
                d, err = None, f"{type(e).__name__}: {e}"[:200]
            dt = time.monotonic() - t0
        pred = ACTION_TO_INTENT.get(d.action) if d else None
        return {
            "id": c["id"], "pred": pred, "latency_s": round(dt, 3), "error": err,
            "decision": d.model_dump() if d else None,
            "args_ok": args_ok(c, d) if d and pred == c["gold"] else (False if c.get("args") else None),
        }

    return await asyncio.gather(*(one(c) for c in cases))


def llm_report(cases, results, label_filter=None, title="") -> None:
    pairs = [(c, r) for c, r in zip(cases, results) if label_filter is None or label_filter(c)]
    n = len(pairs)
    err = sum(1 for _, r in pairs if r["pred"] is None)
    correct = sum(1 for c, r in pairs if r["pred"] == c["gold"])
    print(f"\n── LLM scout{title} — {n} cases ──")
    print(f"  accuracy {pct(correct, n)}   ({correct}/{n})   failed/timeout {err}")
    ap = [r["args_ok"] for _, r in pairs if r["args_ok"] is not None]
    if ap:
        print(f"  argument extraction (widget gold)  {pct(sum(ap), len(ap))}   ({sum(ap)}/{len(ap)}; wrong route counts as wrong)")
    lat = sorted(r["latency_s"] for _, r in pairs if r["pred"] is not None)
    if lat:
        p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]  # noqa: E731
        print(f"  latency  mean {statistics.mean(lat):.2f}s  p50 {p(.5):.2f}s  p90 {p(.9):.2f}s  max {lat[-1]:.2f}s")
    conf: dict[str, Counter] = defaultdict(Counter)
    for c, r in pairs:
        conf[c["gold"]][r["pred"] or "ERROR"] += 1
    cols = INTENTS + ["ERROR"]
    print(f"\n  {'gold \\ pred':<16}" + "".join(f"{k[:11]:>12}" for k in cols))
    for g in INTENTS:
        if conf[g]:
            print(f"  {g:<16}" + "".join(f"{conf[g][k] or '':>12}" for k in cols))


def pipeline_report(cases, kw, llm) -> None:
    """Production behaviour: keyword decision if there is one, else the LLM."""
    rows = []
    for c, k, l in zip(cases, kw, llm):
        final = k["pred"] if k["pred"] is not None else l["pred"]
        rows.append((c, k, l, final))
    n = len(rows)
    ok = sum(1 for c, _, _, f in rows if f == c["gold"])
    llm_ok = sum(1 for c, _, l, _ in rows if l["pred"] == c["gold"])
    skipped = sum(1 for _, k, _, _ in rows if k["pred"] is not None)
    print(f"\n── Production pipeline (keyword, else LLM) — {n} cases ──")
    print(f"  accuracy {pct(ok, n)} ({ok}/{n})   vs LLM alone {pct(llm_ok, n)} ({llm_ok}/{n})")
    print(f"  LLM call skipped on {skipped} cases ({pct(skipped, n)})")
    hurt = [(c, k, l) for c, k, l, f in rows if l["pred"] == c["gold"] and f != c["gold"]]
    helped = [(c, k, l) for c, k, l, f in rows if l["pred"] != c["gold"] and f == c["gold"]]
    print(f"  the keyword shortcut turned a correct LLM answer into a wrong one: {len(hurt)}")
    for c, k, l in hurt:
        print(f"    gold={c['gold']:<15} keyword said {k['pred']:<10} {c['q'][:70]!r}")
    print(f"  ...and fixed a wrong LLM answer: {len(helped)}")
    for c, k, l in helped:
        print(f"    gold={c['gold']:<15} LLM said {str(l['pred']):<15} {c['q'][:70]!r}")
    print("\n── LLM scout: wrong answers ──")
    for c, _, l, _ in rows:
        if l["pred"] != c["gold"]:
            print(f"  gold={c['gold']:<15} pred={str(l['pred']):<15} {c['q'][:70]!r}")



# ── intent router (Decisions API) ───────────────────────────────────────────

SKILL_BARS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


async def run_router(cases: list[dict], concurrency: int, latency_sample: int) -> dict:
    """One Decisions request per case (bulk, `concurrency` in flight), then a short
    sequential pass for the latency a real turn would see."""
    from core import intent_router as ir

    sem = asyncio.Semaphore(concurrency)

    async def one(c: dict) -> dict:
        async with sem:
            for attempt in range(3):  # a beta API: retry transient failures, report persistent ones
                try:
                    r = await ir.route(c["q"])
                    top = max(r.scores, key=r.scores.__getitem__)
                    return {"id": c["id"], "label": r.label, "top": top, "score": round(r.score, 4),
                            "scores": {k: round(x, 4) for k, x in r.scores.items()}}
                except Exception as exc:  # noqa: BLE001
                    err = repr(exc)
                    await asyncio.sleep(1 + attempt)
            return {"id": c["id"], "label": None, "top": None, "score": 0.0, "scores": {}, "error": err}

    t0 = time.monotonic()
    results = await asyncio.gather(*(one(c) for c in cases))
    bulk_s = time.monotonic() - t0

    lat = []
    for c in cases[:: max(1, len(cases) // latency_sample)][:latency_sample]:
        t = time.monotonic()
        try:
            await ir.route(c["q"])
            lat.append(time.monotonic() - t)
        except Exception:  # noqa: BLE001
            pass
    return {"bulk_s": round(bulk_s, 2), "latency_s": [round(x, 3) for x in lat], "results": list(results)}


def router_error_kind(gold: str, pred: str) -> str:
    """Cost-aware label for one router decision (see the module docstring)."""
    if pred == gold:
        return "ok"
    if gold == "about_omni" and pred == "direct_response":
        return "benign"  # no pre-fetch happens; the agent can still load the skill itself
    if gold in WIDGET_INTENTS:
        return "lost-widget"
    if pred in WIDGET_INTENTS:
        return "wrong-widget"
    if pred == "about_omni":
        return "wrong-omni-block"
    if pred.startswith("skill:"):
        return "wrong-skill"  # loads a workflow and a research run nobody asked for
    if gold.startswith("skill:"):
        return "missed-skill"  # benign-ish: the agent can still load the skill itself
    if pred == "web_search":
        return "wasted-search"
    return "lost-enrichment"  # pred direct_response on a query that wanted a lookup


def _router_slice(cases, res, title: str) -> None:
    n = len(cases)
    if not n:
        return
    bad = [r for r in res if r["label"] is None]
    ok = [r["label"] == c["gold"] for c, r in zip(cases, res)]
    cor = sorted(r["score"] for r, o in zip(res, ok) if o)
    wro = sorted(r["score"] for r, o in zip(res, ok) if not o and r["label"])
    med = lambda a: a[len(a) // 2] if a else float("nan")  # noqa: E731
    kinds = Counter(router_error_kind(c["gold"], r["label"]) for c, r in zip(cases, res) if r["label"])
    real = sum(v for k, v in kinds.items() if k not in ("ok", "benign"))
    detail = ", ".join(f"{k} {v}" for k, v in kinds.items() if k != "ok") or "none"
    print(f"\n── Router [{title}] — {n} cases ──")
    print(f"  accuracy: {pct(sum(ok), n)} ({sum(ok)}/{n});  request failures {len(bad)};  "
          f"median probability when right {med(cor):.3f}, when wrong {med(wro):.3f}")
    print(f"  real errors {real}  [{detail}]")
    print(f"  {'gold':<20}{'n':>4}{'right':>9}")
    for g in INTENTS:
        idx = [i for i, c in enumerate(cases) if c["gold"] == g]
        if idx:
            print(f"  {g:<20}{len(idx):>4}{pct(sum(ok[i] for i in idx), len(idx)):>9}")


def router_report(cases, rr: dict) -> None:
    res = rr["results"]
    n = len(cases)
    lat = sorted(rr["latency_s"])
    print(f"\n══ Decisions-API router — {n} cases ══")
    print(f"  {n} requests in {rr['bulk_s']}s")
    if lat:
        p = lambda q: lat[min(len(lat) - 1, int(q * len(lat)))]  # noqa: E731
        print(f"  single-request latency (n={len(lat)}, from this machine, sequential): "
              f"mean {statistics.mean(lat) * 1000:.0f}ms  p50 {p(.5) * 1000:.0f}ms  "
              f"p90 {p(.9) * 1000:.0f}ms  max {lat[-1] * 1000:.0f}ms")
    for title, f in (("all", lambda c: True), ("excluding low-confidence gold", lambda c: "low" not in c["tags"]),
                     ("adversarial only", lambda c: "adv" in c["tags"])):
        idx = [i for i, c in enumerate(cases) if f(c)]
        _router_slice([cases[i] for i in idx], [res[i] for i in idx], title)

    # The skill bar is the one threshold left: sweep it on the cases that involve a skill.
    print(f"\n  skill bar sweep (gold or top choice is a skill; the shipped bar is "
          f"{__import__('core.intent_router', fromlist=['x']).MIN_PROB_SKILL}):")
    print(f"  {'bar':>6}{'skill taken':>13}{'right':>7}{'wrong':>7}")
    rel = [(c, r) for c, r in zip(cases, res) if r["top"] and (c["gold"].startswith("skill:") or r["top"].startswith("skill:"))]
    for t in SKILL_BARS:
        took = [(c, r) for c, r in rel if r["top"].startswith("skill:") and r["scores"][r["top"]] >= t]
        right = sum(1 for c, r in took if c["gold"] == r["top"])
        print(f"  {t:>6.2f}{len(took):>13}{right:>7}{len(took) - right:>7}")


def router_details(cases, rr: dict) -> None:
    print("\n── Router: wrong answers ──")
    for c, r in zip(cases, rr["results"]):
        if r["label"] != c["gold"]:
            kind = router_error_kind(c["gold"], r["label"]) if r["label"] else "REQUEST FAILED " + r.get("error", "")[:60]
            print(f"  p={r['score']:.2f} {kind:<16} gold={c['gold']:<15} pred={str(r['label']):<15} {c['q'][:46]!r}")

# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also run the production scout model (network, ~1 call per case)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=15.0, help="per-call LLM timeout (prod uses 6s; looser here so slow calls are measured, not dropped)")
    ap.add_argument("--router", action="store_true", help="also run the intent router (one Decisions API request per case; needs OPENAI_API_KEY)")
    ap.add_argument("--router-details", action="store_true", help="with --router: list every wrong answer")
    ap.add_argument("--details", action="store_true", help="list every wrong answer and miss for the keyword stage")
    ap.add_argument("--out", type=Path, help="write raw per-case results as JSON")
    a = ap.parse_args()

    cases = load_cases()
    print(f"{len(cases)} cases: " + ", ".join(f"{g} {sum(c['gold'] == g for c in cases)}" for g in INTENTS))
    tagc = Counter(t for c in cases for t in c["tags"])
    print("tags: " + ", ".join(f"{k} {v}" for k, v in sorted(tagc.items())))

    kw = run_keyword(cases)
    keyword_report(cases, kw)
    keyword_report(cases, kw, lambda c: "adv" in c["tags"], " [adversarial only]")
    keyword_report(cases, kw, lambda c: "adv" not in c["tags"] and "long" not in c["tags"], " [non-adversarial, short]")
    keyword_report(cases, kw, lambda c: "low" not in c["tags"], " [excluding low-confidence gold]")
    if a.details:
        keyword_details(cases, kw)

    llm = None
    if a.llm:
        t0 = time.monotonic()
        llm = asyncio.run(run_llm(cases, a.concurrency, a.timeout))
        print(f"\n(LLM run took {time.monotonic() - t0:.0f}s at concurrency {a.concurrency})")
        llm_report(cases, llm)
        llm_report(cases, llm, lambda c: "adv" in c["tags"], " [adversarial only]")
        pipeline_report(cases, kw, llm)

    rr = None
    if a.router:
        rr = asyncio.run(run_router(cases, a.concurrency, latency_sample=30))
        router_report(cases, rr)
        if a.router_details:
            router_details(cases, rr)

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(
            {"cases": cases, "keyword": kw, "llm": llm, "router": rr, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")},
            ensure_ascii=False, indent=1))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
