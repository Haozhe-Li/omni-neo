"""Embedding intent router — a small linear classifier over sentence embeddings.

Replaces the keyword shortcut in core/context_enrichment.py (`shortcut_decision`).
Every example in core/intent_examples.py is embedded once with the shared
multilingual MiniLM, and a softmax-regression head is fitted on those vectors
(numpy only, deterministic, tens of milliseconds at start-up). A query is embedded,
the head gives a probability per label, and the top label is the answer — unless
its probability is below the threshold, in which case there is no decision and the
caller falls through to the LLM scout.

Why a head rather than "label of the nearest anchor": nearest-anchor is captured by
a single topical word ("汇率是怎么形成的" lands on a currency anchor, "what are you
doing this weekend?" on a weather one). The head sees all anchors at once, learns
that question shapes like "how does X work" point to web_search, and on the same
anchors and queries raised top-1 from 71% to 83% (held-out: 58% -> 91%). See
evals/scout_routing for the numbers and `run.py --router` to reproduce them.

Two ways to say "not sure", both returning label=None:

- the top probability is below `MIN_PROB`;
- the query is farther than `MIN_COSINE` from every anchor. A softmax is confident
  about things it has never seen, so far-from-everything queries need their own
  check; it defaults to off until the evals show a value worth enforcing.

Redis is a start-up cache, not part of the request path. Anchor vectors live in
process memory once built; Redis only saves the embedding work on the *next*
start. The key is a hash of the example text and the embedding model, so:

- Editing an example changes the hash, misses the cache and re-embeds — no manual
  invalidation, and nothing stale can be served.
- Restarts and extra instances with unchanged examples load the vectors instead
  of re-embedding on the shared service.
- Keys for superseded example sets are deleted after a successful build. An
  instance still running the old set is unaffected: it holds its vectors in
  memory and never reads Redis again.

`build(force=True)` deletes every cached set first and re-embeds regardless.

Redis being down or stalled only costs the cache (each call is time-bounded): the
router still builds from the embedding service. The embedding service being down is
a real failure and raises — the caller decides what that means (for the scout it
should mean "use the LLM").

`route` is on the critical path of a turn and does one embedding request, so the
caller owns the timeout.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np

from core.intent_examples import INTENT_EXAMPLES, SKILL_LABEL_PREFIX
from core.utils import vector_sources
from core.utils.redis_client import get_async_redis

logger = logging.getLogger(__name__)

# A decision needs the head's probability for its top label to reach that label's
# threshold, otherwise the router says "unknown" (label=None) and the caller asks
# the LLM scout.
#
# web_search gets a higher bar than the rest because it is the catch-all label and
# the only one whose mistakes silently cost something: a stock question taken for a
# lookup loses its stock card. Measured on evals/scout_routing (266 cases): flat 0.6
# decided 73% with 9 real errors (3 lost widgets); 0.6 with web_search at 0.75
# decided 65% with 4 errors and no lost widget. Both numbers were read off the same
# set, so treat 0.75 as a sensible setting rather than a measured optimum. The head
# is under-confident (median probability when right is ~0.8): these are not "75%
# sure" figures. Re-run `run.py --router` before moving them.
MIN_PROB = float(os.getenv("INTENT_ROUTER_MIN_PROB", "0.6"))
MIN_PROB_BY_LABEL: dict[str, float] = {
    "web_search": float(os.getenv("INTENT_ROUTER_MIN_PROB_WEB_SEARCH", "0.75")),
}
# ...and the query must be at least this cosine-close to some anchor. 0 = off: at
# a 0.8 probability bar a floor of 0.5 removed one error and 21 correct decisions,
# which is inside the noise of a set this size.
MIN_COSINE = float(os.getenv("INTENT_ROUTER_MIN_COSINE", "0.0"))
# Skill labels (`skill:<name>`, see core/intent_examples.py) get the highest bar of
# all: a false positive loads a ~12k-char workflow and sends the agent off on a
# multi-step research run for what was a one-shot question, while a miss costs
# nothing — the agent can still load the skill itself. Not yet calibrated: there
# are too few skill cases in evals/scout_routing to read a threshold off, so this
# is a conservative guess; re-run `run.py --router` before lowering it.
MIN_PROB_SKILL = float(os.getenv("INTENT_ROUTER_MIN_PROB_SKILL", "0.8"))


def min_prob_for(label: str) -> float:
    """The probability the head must reach before a decision for `label` stands."""
    if label.startswith(SKILL_LABEL_PREFIX):
        return MIN_PROB_SKILL
    return MIN_PROB_BY_LABEL.get(label, MIN_PROB)


# Softmax-regression head. Full-batch Adam from zero init, so the same anchors give
# the same weights on every instance. C is the inverse L2 strength; picked from
# {1, 10, 100} against evals/scout_routing, so treat it as tuned on that set.
_HEAD_C = 10.0
_HEAD_STEPS = 1500
_HEAD_LR = 0.05

# Overridable so a benchmark can use its own keys and never touch the ones the
# running app reads on start-up (scripts/bench_intent_router.py).
_KEY_PREFIX = os.getenv("INTENT_ROUTER_REDIS_PREFIX", "intent_router:v1:")
# A SET of the vector keys this router has written. Superseded keys are found
# here, never by SCAN/KEYS: those walk the whole keyspace of a Redis shared with
# every checkpoint and cache in the app, and at ~21k keys a cleanup took minutes.
_REGISTRY = _KEY_PREFIX + "registry"
# Redis is only a start-up cache; a stalled call must cost the cache, not the start.
_REDIS_TIMEOUT_S = 5.0
# The service reaches its throughput ceiling at ~4 requests in flight and a deeper
# queue only delays interactive searches sharing it (see core/utils/vector_sources.py).
_BATCH = 32
_CONCURRENCY = 4

Embedder = Callable[[list[str]], Awaitable[list[list[float]]]]


@dataclass(frozen=True)
class RouteResult:
    """The outcome of routing one query.

    ``label`` is None when the router is not sure (see the module docstring).
    ``top_label`` and ``score`` (its probability) are the head's raw answer
    regardless of thresholds, ``scores`` the probability of every label, and
    ``cosine``/``anchor`` the nearest anchor — what a calibration run needs to
    sweep thresholds without re-embedding, and what to look at when a route is
    wrong.
    """

    label: str | None
    top_label: str
    score: float
    anchor: str
    scores: dict[str, float] = field(default_factory=dict)
    cosine: float = 0.0


def _fingerprint(examples: dict[str, list[str]]) -> str:
    blob = json.dumps(
        {"model": vector_sources.DENSE_MODEL, "examples": examples},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.maximum(n, 1e-12)


async def _embed_many(texts: list[str], embed: Embedder) -> np.ndarray:
    """Embed `texts` in batches, a few requests in flight at a time, L2-normalised."""
    sem = asyncio.Semaphore(_CONCURRENCY)

    async def one(batch: list[str]) -> list[list[float]]:
        async with sem:
            return await embed(batch)

    parts = await asyncio.gather(
        *(one(texts[i : i + _BATCH]) for i in range(0, len(texts), _BATCH))
    )
    return _normalize(np.asarray([v for p in parts for v in p], dtype=np.float32))


def _fit_head(X: np.ndarray, y: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial logistic regression with L2, by full-batch Adam.

    X is (n, d) unit rows, y the (n,) class indices. Returns weights (d, k) and
    bias (k,). No randomness anywhere, so every instance fits identical weights.
    """
    X = X.astype(np.float64)
    n, d = X.shape
    Y = np.eye(n_classes)[y]
    W = np.zeros((d + 1, n_classes))
    m = np.zeros_like(W)
    v = np.zeros_like(W)
    for t in range(1, _HEAD_STEPS + 1):
        Z = X @ W[:-1] + W[-1]
        Z -= Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        G = np.vstack([X.T @ (P - Y), (P - Y).sum(axis=0)]) * (_HEAD_C / n)
        G[:-1] += W[:-1] / n
        m = 0.9 * m + 0.1 * G
        v = 0.999 * v + 0.001 * G * G
        W -= _HEAD_LR * (m / (1 - 0.9**t)) / (np.sqrt(v / (1 - 0.999**t)) + 1e-8)
    return W[:-1], W[-1]


class IntentRouter:
    def __init__(
        self,
        labels: list[str],
        texts: list[str],
        vectors: np.ndarray,
        embed: Embedder,
    ) -> None:
        assert len(labels) == len(texts) == len(vectors)
        self._labels = labels
        self._texts = texts
        self._matrix = vectors  # (n_anchors, dim), unit rows
        self._embed = embed
        self.label_set = sorted(set(labels))
        t0 = time.monotonic()
        y = np.array([self.label_set.index(lab) for lab in labels])
        self._W, self._b = _fit_head(vectors, y, len(self.label_set))
        self.fit_seconds = time.monotonic() - t0

    # ── build ───────────────────────────────────────────────────────────────

    @classmethod
    async def build(
        cls,
        examples: dict[str, list[str]] | None = None,
        *,
        force: bool = False,
        embed: Embedder | None = None,
        redis=None,
    ) -> "IntentRouter":
        examples = INTENT_EXAMPLES if examples is None else examples
        for label, items in examples.items():
            if len(items) < 3:
                raise ValueError(f"intent {label!r} needs at least 3 examples, has {len(items)}")
        embed = embed or vector_sources.embed_dense
        redis = redis or get_async_redis()

        labels = [lab for lab, items in examples.items() for _ in items]
        texts = [t for items in examples.values() for t in items]
        key = _KEY_PREFIX + _fingerprint(examples)
        t0 = time.monotonic()

        vectors = None
        if force:
            await cls._cleanup(redis, keep=None, also=key)
        else:
            vectors = await cls._load(redis, key, len(texts))
        source = "redis"
        if vectors is None:
            source = "embedded"
            vectors = await _embed_many(texts, embed)
            await cls._store(redis, key, vectors)
        await cls._cleanup(redis, keep=key)

        logger.info(
            "[intent_router] %d anchors / %d labels ready in %.2fs (%s)",
            len(texts), len(set(labels)), time.monotonic() - t0, source,
        )
        router = cls(labels, texts, vectors, embed)
        logger.info("[intent_router] head fitted in %.0fms", router.fit_seconds * 1000)
        return router

    @staticmethod
    async def _load(redis, key: str, n: int) -> np.ndarray | None:
        try:
            raw = await asyncio.wait_for(redis.get(key), _REDIS_TIMEOUT_S)
            if not raw:
                return None
            p = json.loads(raw)
            v = np.frombuffer(base64.b64decode(p["vectors"]), dtype=np.float32)
            v = v.reshape(p["n"], p["dim"])
            if p["model"] != vector_sources.DENSE_MODEL or p["n"] != n:
                return None
            return v
        except Exception as exc:  # corrupt, unreadable or slow cache is just a miss
            logger.warning("[intent_router] cache read failed (%r) — re-embedding", exc)
            return None

    @staticmethod
    async def _store(redis, key: str, vectors: np.ndarray) -> None:
        payload = {
            "model": vector_sources.DENSE_MODEL,
            "n": int(vectors.shape[0]),
            "dim": int(vectors.shape[1]),
            "vectors": base64.b64encode(vectors.astype(np.float32).tobytes()).decode("ascii"),
        }
        try:
            await asyncio.wait_for(redis.set(key, json.dumps(payload)), _REDIS_TIMEOUT_S)
        except Exception as exc:
            logger.warning("[intent_router] cache write failed (%r) — continuing without it", exc)

    @staticmethod
    async def _cleanup(redis, keep: str | None, also: str | None = None) -> None:
        """Delete every registered vector key except `keep` (and `also`, which may
        predate the registry), then register `keep`."""
        try:
            async def work() -> None:
                stale = {k for k in await redis.smembers(_REGISTRY) if k != keep}
                if also and also != keep:
                    stale.add(also)
                if stale:
                    await redis.delete(*stale)
                    await redis.srem(_REGISTRY, *stale)
                if keep:
                    await redis.sadd(_REGISTRY, keep)

            await asyncio.wait_for(work(), _REDIS_TIMEOUT_S)
        except Exception as exc:
            logger.warning("[intent_router] cache cleanup failed (%r)", exc)

    # ── route ───────────────────────────────────────────────────────────────

    async def embed_queries(self, queries: list[str]) -> np.ndarray:
        """Normalised embeddings for many queries at once (calibration runs)."""
        return await _embed_many(queries, self._embed)

    def route_vector(
        self,
        q: np.ndarray,
        min_prob: float | None = None,
        min_cosine: float | None = None,
    ) -> RouteResult:
        """Route an already-embedded, unit-length query.

        `min_prob` overrides the per-label thresholds with one flat value (that is
        how a calibration run sweeps them); None uses `min_prob_for`.
        """
        c_min = MIN_COSINE if min_cosine is None else min_cosine
        z = q.astype(np.float64) @ self._W + self._b
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        top = int(np.argmax(p))
        sims = self._matrix @ q
        nearest = int(np.argmax(sims))
        score, cosine = float(p[top]), float(sims[nearest])
        p_min = min_prob_for(self.label_set[top]) if min_prob is None else min_prob
        return RouteResult(
            label=self.label_set[top] if score >= p_min and cosine >= c_min else None,
            top_label=self.label_set[top],
            score=score,
            anchor=self._texts[nearest],
            scores={lab: float(x) for lab, x in zip(self.label_set, p)},
            cosine=cosine,
        )

    async def route(
        self, query: str, min_prob: float | None = None, min_cosine: float | None = None
    ) -> RouteResult:
        q = (await self.embed_queries([query]))[0]
        return self.route_vector(q, min_prob, min_cosine)


# ── process-wide router ─────────────────────────────────────────────────────
# Building takes 1-2s (Redis read or a full embed), which no user request may pay.
# The app builds it once at start-up (`warm_intent_router`, from the lifespan hook)
# and requests use `get_ready_router`, which never waits: it returns the router if
# it is built, otherwise None, so the caller uses the LLM scout. If start-up could
# not build it (embedding service down) a later request retries in the background,
# at most once per `_RETRY_COOLDOWN_S`.

_router: IntentRouter | None = None
_build_task: asyncio.Task | None = None
_last_failure = float("-inf")
_RETRY_COOLDOWN_S = 30.0


async def _build_once() -> IntentRouter | None:
    global _router, _last_failure
    try:
        _router = await IntentRouter.build()
    except Exception as exc:
        _last_failure = time.monotonic()
        logger.warning("[intent_router] build failed (%r) — the LLM scout handles every turn until it succeeds", exc)
    return _router


async def warm_intent_router(timeout: float = 20.0) -> None:
    """Build the process-wide router; never raises, never waits past `timeout`."""
    global _build_task
    if _router is not None:
        return
    if _build_task is None or _build_task.done():
        _build_task = asyncio.create_task(_build_once())
    try:
        await asyncio.wait_for(asyncio.shield(_build_task), timeout)
    except Exception as exc:  # timeout or failure: the build task carries on / has logged
        logger.warning("[intent_router] not ready after %.0fs (%r) — continuing without it", timeout, exc)


def get_ready_router() -> IntentRouter | None:
    """The router if it is built; otherwise None, and a background build is
    started (rate-limited) so a later turn can use it."""
    global _build_task
    if _router is not None:
        return _router
    if (_build_task is None or _build_task.done()) and time.monotonic() - _last_failure > _RETRY_COOLDOWN_S:
        try:
            _build_task = asyncio.get_running_loop().create_task(_build_once())
        except RuntimeError:  # no running loop (called from sync code): nothing to build on
            pass
    return None
