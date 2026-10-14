"""The pre-model prelude of a turn, offline: the prompt guard running alongside the
scout, and a thread the client named itself being registered on first use.

    venv/bin/python3.12 -m pytest tests/test_turn_prelude.py -q

The contract under test:

- the guard overlaps the scout (latency is the max of the two, not the sum);
- a harmful verdict ends the turn before anything is emitted and before the
  model is called, whatever the scout was doing;
- the scout's tool fetch (the first step with an outside effect) waits for the
  verdict and is skipped when the query was flagged;
- /chat and /sync register an unclaimed UUID thread idempotently, and never
  touch a thread someone else owns.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import dotenv
import pytest

dotenv.load_dotenv()
for _key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY", "WANDB_API_KEY"):
    os.environ.setdefault(_key, "test-placeholder")

from core import context_enrichment as ce  # noqa: E402
from core import stream as stream_mod  # noqa: E402
from core.context_enrichment import Enrichment  # noqa: E402
from core.database import db_threads_control as dtc  # noqa: E402


def run(coro):
    return asyncio.run(coro)


# ── guard ‖ scout ──────────────────────────────────────────────────────────


class FakeAgent:
    def __init__(self):
        self.started = 0

    async def astream(self, *a, **kw):
        self.started += 1
        return
        yield  # pragma: no cover — an async generator that yields nothing


@pytest.fixture
def prelude(monkeypatch):
    """Patch everything `_stream_agent` touches before the model, recording when
    the scout and the guard start/finish."""
    log: dict = {"agent": FakeAgent(), "marks": {}, "locked": []}

    def install(*, guard_delay, guard_verdict, scout_delay, scout_text="CTX"):
        async def fake_guard(_q):
            log["marks"]["guard_start"] = time.monotonic()
            await asyncio.sleep(guard_delay)
            log["marks"]["guard_end"] = time.monotonic()
            return guard_verdict

        async def fake_enrich(query, **kw):
            log["marks"]["scout_start"] = time.monotonic()
            await asyncio.sleep(scout_delay)
            log["marks"]["scout_end"] = time.monotonic()
            return Enrichment(
                action="web_search", text=scout_text,
                events=[{"type": "tool_call", "tool": "web_search", "args": {}}],
            )

        async def fake_reset(*a, **kw):
            return None

        monkeypatch.setattr(stream_mod, "is_harmful", fake_guard)
        monkeypatch.setattr(stream_mod, "enrich_context", fake_enrich)
        monkeypatch.setattr(stream_mod, "reset_citation_registry_async", fake_reset)
        monkeypatch.setattr(stream_mod, "get_agent", lambda _m: log["agent"])
        monkeypatch.setattr(stream_mod, "_lock_thread_fire_and_forget", lambda tid: log["locked"].append(tid))

    log["install"] = install
    return log


def collect(**kw) -> list[dict]:
    async def go():
        out = []
        async for ev in stream_mod.run_agent_stream(
            query="hello", thread_id="t1", model_id="best", turn=1, **kw
        ):
            out.append(json.loads(ev[6:]))
        return out

    return run(go())


def test_guard_overlaps_the_scout_instead_of_adding_to_it(prelude):
    prelude["install"](guard_delay=0.15, guard_verdict=False, scout_delay=0.15)
    t0 = time.monotonic()
    collect()
    elapsed = time.monotonic() - t0
    m = prelude["marks"]
    assert prelude["agent"].started == 1
    # The scout began before the guard answered — not after.
    assert m["scout_start"] < m["guard_end"]
    # ~0.15s overlapped, not ~0.30s back to back.
    assert elapsed < 0.27, elapsed


def test_harmful_verdict_stops_everything_before_anything_is_emitted(prelude):
    prelude["install"](guard_delay=0.05, guard_verdict=True, scout_delay=0.01)
    events = collect()
    assert prelude["agent"].started == 0, "the model must never see a flagged query"
    assert prelude["locked"] == ["t1"]
    # Only the safety error: no scout tool_call, no sources, no done.
    assert [e["type"] for e in events] == ["error"]
    assert events[0].get("code") == "safety_terminated"


def test_harmful_verdict_after_a_slow_scout_still_blocks_the_model(prelude):
    # The scout outlasts the guard: the gate must still hold.
    prelude["install"](guard_delay=0.01, guard_verdict=True, scout_delay=0.1)
    events = collect()
    assert prelude["agent"].started == 0
    assert [e["type"] for e in events] == ["error"]


def test_benign_turn_reaches_the_model_and_emits_the_scout_step(prelude):
    prelude["install"](guard_delay=0.01, guard_verdict=False, scout_delay=0.01)
    events = collect()
    assert prelude["agent"].started == 1
    assert events[0]["type"] == "tool_call"


def test_unfinished_guard_is_cancelled_when_the_stream_is_abandoned(prelude, monkeypatch):
    prelude["install"](guard_delay=5.0, guard_verdict=False, scout_delay=0.0)
    cancelled = []

    async def slow_guard(_q):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return False

    monkeypatch.setattr(stream_mod, "is_harmful", slow_guard)

    async def go():
        gen = stream_mod.run_agent_stream(query="hi", thread_id="t1", model_id="best", turn=1)
        task = asyncio.ensure_future(gen.__anext__())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await gen.aclose()
        await asyncio.sleep(0)

    run(go())
    assert cancelled == [True]


# ── enrich_context: the tool fetch waits for the verdict ───────────────────


def _patch_scout(monkeypatch, ran: list):
    class R:
        async def route(self, q):
            from core.intent_router import RouteResult
            return RouteResult(label="web_search", top_label="web_search", score=0.9, anchor="a")

    monkeypatch.setattr(ce, "get_ready_router", lambda: R())

    async def fake_run(d):
        ran.append(d)
        return "BODY", None

    real = ce._ACTIONS["web_search"]
    monkeypatch.setitem(
        ce._ACTIONS, "web_search",
        ce._Action(tool=real.tool, args=real.args, run=fake_run, ready=real.ready),
    )


def test_flagged_query_never_reaches_the_scouts_tool(monkeypatch):
    ran: list = []
    _patch_scout(monkeypatch, ran)

    async def go():
        verdict = asyncio.get_running_loop().create_future()
        verdict.set_result(True)
        return await ce.enrich_context("what is a vector database", harmful=verdict)

    out = run(go())
    assert ran == [] and out.text == "" and out.events == []


def test_cleared_query_still_gets_its_tool_fetch(monkeypatch):
    ran: list = []
    _patch_scout(monkeypatch, ran)

    async def go():
        verdict = asyncio.get_running_loop().create_future()
        verdict.set_result(False)
        return await ce.enrich_context("what is a vector database", harmful=verdict)

    out = run(go())
    assert len(ran) == 1 and "BODY" in out.text


# ── lazy thread registration ───────────────────────────────────────────────

UUID = "123e4567-e89b-42d3-a456-426614174000"


def test_only_well_formed_uuids_are_registered_lazily():
    assert dtc.is_uuid_thread_id(UUID)
    assert dtc.is_uuid_thread_id(UUID.upper())
    for bad in ("", None, "local-123-abc", "mock-thread-id-1", UUID + "x", "../etc/passwd"):
        assert not dtc.is_uuid_thread_id(bad)


def test_ensure_thread_registered_creates_rows_and_drops_the_stale_unclaimed_cache(monkeypatch):
    calls = []
    monkeypatch.setattr(dtc, "register_thread", lambda tid, uid: calls.append((tid, uid)) or True)
    dtc._owner_cache_put(UUID, None)  # what the /chat owner read cached for a new thread
    assert dtc.owner_cache_hit(UUID)
    assert dtc.ensure_thread_registered(UUID, "user_1") is True
    assert calls == [(UUID, "user_1")]
    assert not dtc.owner_cache_hit(UUID), "a cached 'unclaimed' owner must not outlive the registration"


def test_sync_registers_an_unclaimed_thread_before_writing(monkeypatch):
    from core.routers import threads

    order = []
    monkeypatch.setattr(threads, "get_thread_owner", lambda tid: None)
    monkeypatch.setattr(threads, "ensure_thread_registered", lambda tid, uid: order.append("register"))
    monkeypatch.setattr(
        threads, "upsert_thread_messages", lambda tid, uid, msgs: order.append("upsert") or True
    )
    out = threads.api_sync_thread(UUID, threads.SyncThreadRequest(messages=[]), user_id="user_1")
    assert out == {"status": "success"}
    assert order == ["register", "upsert"]


def test_sync_leaves_an_owned_thread_alone(monkeypatch):
    from core.routers import threads

    order = []
    monkeypatch.setattr(threads, "get_thread_owner", lambda tid: "someone_else")
    monkeypatch.setattr(threads, "ensure_thread_registered", lambda tid, uid: order.append("register"))
    monkeypatch.setattr(
        threads, "upsert_thread_messages", lambda tid, uid, msgs: order.append("upsert") or False
    )
    with pytest.raises(Exception):  # 404: the ownership-guarded upsert touched no row
        threads.api_sync_thread(UUID, threads.SyncThreadRequest(messages=[]), user_id="user_1")
    assert order == ["upsert"]


def test_background_job_registers_only_new_threads(monkeypatch):
    from core.routers import chat

    seen = []
    monkeypatch.setattr(chat, "ensure_thread_registered", lambda tid, uid: seen.append("register"))
    monkeypatch.setattr(chat, "touch_thread", lambda tid, uid: seen.append("touch"))
    chat._register_and_touch(UUID, "u", True)
    chat._register_and_touch(UUID, "u", False)
    assert seen == ["register", "touch", "touch"]
