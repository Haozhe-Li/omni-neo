"""The `best` router (core/routing.py): which model serves a turn, and what it costs.

    python3 -m pytest tests/test_routing.py -q
"""
from __future__ import annotations

import asyncio
import os

import pytest

for _key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY", "WANDB_API_KEY"):
    os.environ.setdefault(_key, "test-placeholder")

from core import routing  # noqa: E402
from core.chat_models import CHAT_MODELS, resolve_model  # noqa: E402
from core.database import db_user_usage as usage  # noqa: E402
from core.routing import decide_route, exchange_number, pick, remember_escalation  # noqa: E402


def run(coro):
    return asyncio.run(coro)


class FakeRedis:
    def __init__(self):
        self.data: dict[str, int] = {}
        self.sets: list[tuple] = []

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        self.sets.append((key, value, ex))
        self.data[key] = value


@pytest.fixture
def redis(monkeypatch):
    r = FakeRedis()
    monkeypatch.setattr(routing, "get_async_redis", lambda: r)
    return r


BEST = CHAT_MODELS["best"]


# The frontend sends the index its reply lands at: 1, 3, 5, … for exchanges 1, 2, 3, …
@pytest.mark.parametrize("turn,exchange", [(None, None), (0, None), (1, 1), (2, 1), (3, 2), (9, 5), (11, 6), (13, 7)])
def test_exchange_number(turn, exchange):
    assert exchange_number(turn) == exchange


def test_the_first_five_exchanges_are_rix_and_the_sixth_onward_luna():
    assert [pick(n, None, None)[0] for n in range(1, 8)] == ["rix"] * 5 + ["luna"] * 2


@pytest.mark.parametrize("skill", ["web-research", "trip-advisor", "guided-learning"])
def test_an_explicit_skill_is_luna_from_the_first_exchange(skill):
    assert pick(1, skill, None) == ("luna", "skill")


@pytest.mark.parametrize("skill", [None, "charting", "report-writing", "draft-email"])
def test_other_skills_do_not_escalate(skill):
    assert pick(1, skill, None)[0] == "rix"


def test_an_escalated_thread_stays_on_luna_from_that_exchange_on():
    assert pick(3, None, 2) == ("luna", "escalated")
    assert pick(2, None, 2) == ("luna", "escalated")
    assert pick(1, None, 2)[0] == "rix"  # rewound to before the skill turn
    assert pick(None, None, 2)[0] == "luna"  # unknown exchange: trust the flag


def test_an_unknown_exchange_with_no_flag_is_rix():
    assert pick(None, None, None) == ("rix", "early")


def test_explicit_models_are_never_routed(redis):
    for k in ("rix", "luna", "gemini"):
        r = run(decide_route(resolve_model(k), "t1", 99, "web-research"))
        assert (r.model_id, r.reason, r.escalate_from) == (k, "explicit", None)


def test_best_routes_by_turn(redis):
    assert run(decide_route(BEST, "t1", 9, None)).model_id == "rix"     # exchange 5
    assert run(decide_route(BEST, "t1", 11, None)).model_id == "luna"   # exchange 6


def test_a_skill_turn_escalates_the_thread_but_only_once_committed(redis):
    r = run(decide_route(BEST, "t1", 3, "web-research"))  # exchange 2
    assert (r.model_id, r.reason, r.escalate_from) == ("luna", "skill", 2)
    assert redis.sets == []  # deciding writes nothing: the usage gate may still refuse
    run(remember_escalation("t1", r))
    assert redis.data == {"route_escalated:t1": 2}
    # The next exchange has no skill, but the thread is already on luna.
    nxt = run(decide_route(BEST, "t1", 5, None))
    assert (nxt.model_id, nxt.reason, nxt.escalate_from) == ("luna", "escalated", None)
    # Other threads are unaffected.
    assert run(decide_route(BEST, "t2", 5, None)).model_id == "rix"


def test_a_repeated_skill_turn_does_not_rewrite_the_flag(redis):
    redis.data["route_escalated:t1"] = 2
    r = run(decide_route(BEST, "t1", 5, "web-research"))
    assert r.model_id == "luna" and r.escalate_from is None
    run(remember_escalation("t1", r))
    assert redis.sets == []


def test_a_skill_on_an_earlier_exchange_moves_the_flag_back(redis):
    redis.data["route_escalated:t1"] = 4
    r = run(decide_route(BEST, "t1", 3, "trip-advisor"))  # exchange 2 < 4
    assert r.escalate_from == 2


def test_no_thread_id_still_routes_but_remembers_nothing(redis):
    r = run(decide_route(BEST, None, 1, "guided-learning"))
    assert r.model_id == "luna"
    run(remember_escalation(None, r))
    assert redis.sets == []


def test_redis_failure_degrades_to_the_free_model(monkeypatch):
    class Broken:
        async def get(self, key):
            raise ConnectionError("down")

        async def set(self, *a, **k):
            raise ConnectionError("down")

    monkeypatch.setattr(routing, "get_async_redis", lambda: Broken())
    r = run(decide_route(BEST, "t1", 3, None))
    assert r.model_id == "rix"
    run(remember_escalation("t1", routing.Route("luna", "skill", 2)))  # must not raise


def test_every_route_target_is_a_concrete_model():
    for target in (routing.EARLY_ROUTE, routing.LATE_ROUTE):
        assert target in CHAT_MODELS and not CHAT_MODELS[target].is_router


# ── billing ────────────────────────────────────────────────────────────────

def test_rix_costs_nothing_and_luna_three():
    assert usage.MODE_CREDIT_COST["rix"] == 0.0
    assert usage.MODE_CREDIT_COST["luna"] == 3.0


def test_a_free_turn_passes_the_gate_even_when_out_of_credits():
    ctx = usage._charge_context("guest_x", "rix")
    row = {"day": ctx["today_iso"], "day_used": ctx["day_limit"], "month": ctx["month"],
           "month_used": ctx["month_limit"], "extra_granted": 0, "extra_used": 0}
    write, result = usage._charge_decision("guest_x", row, ctx)
    assert result["charged"] is True and write["day_used"] == ctx["day_limit"]
    # ...while a paid turn on the same exhausted account is refused.
    ctx = usage._charge_context("guest_x", "luna")
    write, result = usage._charge_decision("guest_x", row, ctx)
    assert result["charged"] is False and write is None


def test_a_free_turn_neither_reads_nor_writes_the_ledger(monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("ledger touched")

    monkeypatch.setattr(usage, "evaluate_charge_async", boom)
    monkeypatch.setattr(usage, "charge_credits_async", boom)
    assert run(usage.evaluate_charge_fast("guest_x", "rix"))["charged"] is True

    async def go():
        usage.commit_charge_fast("guest_x", "rix")  # would need a running loop if it scheduled work
        await asyncio.sleep(0)

    run(go())
