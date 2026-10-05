"""Intent router logic, offline: fake embedder, fake Redis, no network.

    venv/bin/python3.12 -m pytest tests/test_intent_router.py -q

What this can and cannot show: it pins the *mechanics* (cache keying, stale-key
cleanup, force rebuild, Redis outage, batching bound, thresholding). It says nothing
about whether MiniLM separates the real intents — that is evals/scout_routing
(`run.py --router`), which needs the live embedding service.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

from core import intent_router as ir  # noqa: E402
from core.intent_examples import INTENT_EXAMPLES  # noqa: E402

DIM = 384


def _vec(text: str) -> list[float]:
    """Deterministic bag-of-character-trigrams embedding: identical strings get
    identical vectors, strings sharing trigrams get a high cosine."""
    v = np.zeros(DIM, dtype=np.float32)
    t = f"  {text.lower()}  "
    for i in range(len(t) - 2):
        h = int(hashlib.md5(t[i : i + 3].encode()).hexdigest(), 16)
        v[h % DIM] += 1.0
    return v.tolist()


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0
        self.texts = 0
        self.in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts += len(texts)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.01)
        self.in_flight -= 1
        return [_vec(t) for t in texts]


class FakeRedis:
    """The four calls the router makes, with the real client's str semantics."""

    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}
        self.forbidden: list[str] = []

    async def get(self, k):
        return self.data.get(k)

    async def set(self, k, v):
        self.data[k] = v

    async def delete(self, *ks):
        for k in ks:
            self.data.pop(k, None)

    async def sadd(self, k, *vs):
        self.sets.setdefault(k, set()).update(vs)

    async def smembers(self, k):
        return set(self.sets.get(k, ()))

    async def srem(self, k, *vs):
        self.sets.get(k, set()).difference_update(vs)

    # Walk the whole shared keyspace: must never be called (see _REGISTRY).
    def scan_iter(self, *a, **kw):
        self.forbidden.append("scan_iter")
        raise AssertionError("SCAN is O(keyspace)")

    async def keys(self, *a, **kw):
        self.forbidden.append("keys")
        raise AssertionError("KEYS is O(keyspace)")

    def vector_keys(self) -> set[str]:
        return {k for k in self.data if k.startswith(ir._KEY_PREFIX)}


class DeadRedis:
    async def get(self, k):
        raise ConnectionError("down")

    set = delete = sadd = smembers = srem = get


class HungRedis:
    """Accepts the connection and never answers."""

    async def _hang(self, *a, **kw):
        await asyncio.sleep(3600)

    get = set = delete = sadd = smembers = srem = _hang


EX = {
    "weather": ["weather in paris", "will it rain tomorrow", "temperature today"],
    "stock": ["nvidia stock price", "tesla shares today", "apple market cap"],
}


def run(coro):
    return asyncio.run(coro)


def test_rejects_label_with_fewer_than_three_examples():
    with pytest.raises(ValueError, match="at least 3"):
        run(ir.IntentRouter.build({"a": ["x", "y"]}, embed=FakeEmbedder(), redis=FakeRedis()))


def test_shipped_examples_are_well_formed():
    assert set(INTENT_EXAMPLES) == {
        "about_omni", "weather", "stock", "currency", "web_search", "direct_response",
        "skill:web-research",
    }
    seen: dict[str, str] = {}
    for label, items in INTENT_EXAMPLES.items():
        assert len(items) >= 3
        for t in items:
            key = t.strip().lower()
            assert key not in seen, f"{t!r} appears under both {seen[key]!r} and {label!r}"
            seen[key] = label


def test_second_build_with_same_examples_reads_cache_and_does_not_embed():
    redis, emb = FakeRedis(), FakeEmbedder()
    run(ir.IntentRouter.build(EX, embed=emb, redis=redis))
    first = emb.texts
    assert first == 6 and len(redis.vector_keys()) == 1
    r2 = run(ir.IntentRouter.build(EX, embed=emb, redis=redis))
    assert emb.texts == first  # nothing embedded the second time
    assert run(r2.route("will it rain tomorrow", min_prob=0.5)).label == "weather"


def test_edited_examples_reembed_and_prune_the_old_key():
    redis, emb = FakeRedis(), FakeEmbedder()
    run(ir.IntentRouter.build(EX, embed=emb, redis=redis))
    (old_key,) = redis.vector_keys()
    edited = {**EX, "weather": [*EX["weather"], "is it snowing"]}
    run(ir.IntentRouter.build(edited, embed=emb, redis=redis))
    assert len(redis.vector_keys()) == 1 and old_key not in redis.data
    assert redis.sets[ir._REGISTRY] == redis.vector_keys()  # registry tracks exactly what exists
    assert emb.texts == 6 + 7
    assert not redis.forbidden


def test_force_deletes_everything_and_reembeds():
    redis, emb = FakeRedis(), FakeEmbedder()
    run(ir.IntentRouter.build(EX, embed=emb, redis=redis))
    redis.data["intent_router:v1:leftover"] = "x"
    redis.sets[ir._REGISTRY].add("intent_router:v1:leftover")
    redis.data["unrelated:key"] = "keep me"
    run(ir.IntentRouter.build(EX, force=True, embed=emb, redis=redis))
    assert emb.texts == 12
    assert "intent_router:v1:leftover" not in redis.data
    assert redis.data["unrelated:key"] == "keep me"  # only keys we registered are touched
    assert len(redis.vector_keys()) == 1 and not redis.forbidden


def test_corrupt_cache_entry_is_a_miss_not_a_crash():
    redis, emb = FakeRedis(), FakeEmbedder()
    run(ir.IntentRouter.build(EX, embed=emb, redis=redis))
    (key,) = redis.vector_keys()
    redis.data[key] = "{not json"
    run(ir.IntentRouter.build(EX, embed=emb, redis=redis))
    assert emb.texts == 12


def test_redis_outage_only_costs_the_cache():
    emb = FakeEmbedder()
    r = run(ir.IntentRouter.build(EX, embed=emb, redis=DeadRedis()))
    assert run(r.route("nvidia stock price", min_prob=0.5)).label == "stock"


def test_stalled_redis_costs_the_cache_not_the_start(monkeypatch):
    monkeypatch.setattr(ir, "_REDIS_TIMEOUT_S", 0.05)
    emb = FakeEmbedder()
    r = run(ir.IntentRouter.build(EX, embed=emb, redis=HungRedis()))
    assert run(r.route("tesla shares today", min_prob=0.5)).label == "stock"


def test_probability_threshold_controls_label_but_not_the_raw_answer():
    r = run(ir.IntentRouter.build(EX, embed=FakeEmbedder(), redis=FakeRedis()))
    exact = run(r.route("weather in paris", min_prob=0.5))
    assert exact.label == "weather" and exact.top_label == "weather"
    assert exact.anchor == "weather in paris" and exact.cosine == pytest.approx(1.0, abs=1e-5)
    assert exact.score > 0.5

    # No probability reaches 1.01: no decision, but the raw answer is still reported.
    unsure = run(r.route("weather in paris", min_prob=1.01))
    assert unsure.label is None and unsure.top_label == "weather"

    assert set(unsure.scores) == set(EX)                       # a probability per label...
    assert sum(unsure.scores.values()) == pytest.approx(1.0)   # ...that is a distribution
    assert max(unsure.scores, key=unsure.scores.get) == unsure.top_label


def test_cosine_floor_catches_queries_far_from_every_anchor():
    r = run(ir.IntentRouter.build(EX, embed=FakeEmbedder(), redis=FakeRedis()))
    far = run(r.route("zzzz qqqq", min_prob=0.0, min_cosine=-1.0))
    assert far.label is not None                               # softmax always picks something...
    guarded = run(r.route("zzzz qqqq", min_prob=0.0, min_cosine=0.9))
    assert guarded.label is None and guarded.cosine < 0.9      # ...the floor is what says "unknown"


def test_head_separates_the_anchors_it_was_trained_on():
    r = run(ir.IntentRouter.build(INTENT_EXAMPLES, embed=FakeEmbedder(), redis=FakeRedis()))
    wrong = [
        (lab, t) for lab, items in INTENT_EXAMPLES.items() for t in items
        if run(r.route(t, min_prob=0.0)).top_label != lab
    ]
    assert not wrong, wrong


def test_head_is_deterministic_across_builds():
    a = run(ir.IntentRouter.build(EX, embed=FakeEmbedder(), redis=FakeRedis()))
    b = run(ir.IntentRouter.build(EX, embed=FakeEmbedder(), redis=FakeRedis()))
    assert np.array_equal(a._W, b._W) and np.array_equal(a._b, b._b)


def test_fit_head_learns_a_linearly_separable_problem_and_starts_fast():
    rng = np.random.default_rng(0)
    centres = np.eye(DIM)[:3] * 3
    X = np.vstack([c + rng.normal(0, 0.3, (20, DIM)) for c in centres])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.repeat(np.arange(3), 20)
    t0 = time.monotonic()
    W, b = ir._fit_head(X, y, 3)
    assert time.monotonic() - t0 < 2.0
    assert ((X @ W + b).argmax(1) == y).all()


def test_embedding_in_batches_is_concurrent_but_bounded():
    many = {"a": [f"alpha {i}" for i in range(300)], "b": [f"beta {i}" for i in range(3)]}
    emb = FakeEmbedder()
    run(ir.IntentRouter.build(many, embed=emb, redis=FakeRedis()))
    assert emb.calls == 10  # ceil(303 / 32)
    assert 1 < emb.max_in_flight <= ir._CONCURRENCY


def test_per_label_thresholds_apply_when_no_flat_value_is_given(monkeypatch):
    r = run(ir.IntentRouter.build(EX, embed=FakeEmbedder(), redis=FakeRedis()))
    monkeypatch.setattr(ir, "MIN_PROB", 0.0)
    monkeypatch.setattr(ir, "MIN_PROB_BY_LABEL", {"weather": 1.01})  # weather can never clear its bar
    assert run(r.route("nvidia stock price")).label == "stock"
    blocked = run(r.route("weather in paris"))
    assert blocked.label is None and blocked.top_label == "weather"
    # an explicit flat value overrides the per-label table (how a sweep calls it)
    assert run(r.route("weather in paris", min_prob=0.0)).label == "weather"


def test_shipped_defaults_hold_web_search_to_a_higher_bar():
    assert ir.min_prob_for("web_search") > ir.min_prob_for("weather") == ir.MIN_PROB


def _fresh_module_state(monkeypatch):
    monkeypatch.setattr(ir, "_router", None)
    monkeypatch.setattr(ir, "_build_task", None)
    monkeypatch.setattr(ir, "_last_failure", float("-inf"))


def test_warm_builds_once_and_requests_get_the_router_without_waiting(monkeypatch):
    _fresh_module_state(monkeypatch)
    built = []

    async def fake_build(cls, *a, **kw):
        built.append(1)
        await asyncio.sleep(0.01)
        return "ROUTER"

    monkeypatch.setattr(ir.IntentRouter, "build", classmethod(fake_build))

    async def go():
        assert ir.get_ready_router() is None       # not built yet: returns at once, kicks off a build
        await ir.warm_intent_router()              # joins that same build rather than starting another
        assert ir.get_ready_router() == "ROUTER"
        await ir.warm_intent_router()              # already built: no second build
    run(go())
    assert len(built) == 1 and ir._router == "ROUTER"


def test_failed_build_never_raises_and_retries_are_rate_limited(monkeypatch):
    _fresh_module_state(monkeypatch)
    calls = []

    async def failing_build(cls, *a, **kw):
        calls.append(1)
        raise RuntimeError("embedding service down")

    monkeypatch.setattr(ir.IntentRouter, "build", classmethod(failing_build))

    async def go():
        await ir.warm_intent_router()          # start-up: logs, does not raise
        assert ir.get_ready_router() is None   # inside the cooldown: no new attempt
        assert ir.get_ready_router() is None
        await asyncio.sleep(0.01)
    run(go())
    assert len(calls) == 1

    monkeypatch.setattr(ir, "_RETRY_COOLDOWN_S", 0.0)  # cooldown over: next request retries

    async def retry():
        ir.get_ready_router()
        await asyncio.sleep(0.05)
    run(retry())
    assert len(calls) == 2


def test_warm_gives_up_waiting_after_its_timeout_but_the_build_continues(monkeypatch):
    _fresh_module_state(monkeypatch)

    async def slow_build(cls, *a, **kw):
        await asyncio.sleep(0.3)
        return "ROUTER"

    monkeypatch.setattr(ir.IntentRouter, "build", classmethod(slow_build))

    async def go():
        t0 = time.monotonic()
        await ir.warm_intent_router(timeout=0.05)   # app start is not held hostage
        assert time.monotonic() - t0 < 0.2
        assert ir.get_ready_router() is None
        await asyncio.sleep(0.4)
        assert ir.get_ready_router() == "ROUTER"    # ...and it landed in the background
    run(go())
