"""The Decisions-API intent router: what it asks, how it reads the answer, how it fails.

    venv/bin/python3.12 -m pytest tests/test_intent_router.py -q
"""
import asyncio
import json

import httpx
import pytest

from core import intent_router as ir
from core.intent_choices import INSTRUCTIONS, INTENT_CHOICES, SKILL_LABEL_PREFIX


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    # Pinned rather than setdefault: another test module may have loaded the real .env.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def answer(top, p_top=0.9, **others):
    """A Decisions response body whose probabilities put `p_top` on `top` and
    spread the rest over the other options (or use `others` for exact values)."""
    labels = list(INTENT_CHOICES)
    rest = (1 - p_top) / (len(labels) - 1)
    probs = {lab: others.get(lab, rest) for lab in labels}
    probs[top] = p_top
    return {"answers": [{
        "type": "choice", "name": "intent", "choice": top,
        "probabilities": [{"value": k, "probability": v} for k, v in probs.items()],
        "confidence": p_top,
    }]}


def client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── the question that is asked ──────────────────────────────────────────────


def test_request_is_one_choice_question_over_every_option():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=answer("weather"))

    run(ir.route("东京明天下雨吗", client=client_for(handler)))
    body = seen["body"]
    assert seen["auth"] == "Bearer test-key"
    assert body["model"] == "gpt-6-luna" and body["input"] == "东京明天下雨吗"
    (q,) = body["questions"]
    assert q["type"] == "choice" and q["name"] == "intent" and q["instructions"] == INSTRUCTIONS
    assert {c["value"]: c["description"] for c in q["choices"]} == INTENT_CHOICES


def test_the_options_are_the_scouts_kinds_plus_the_skill():
    assert set(INTENT_CHOICES) == {
        "weather", "stock", "currency", "about_omni", "web_search", "direct_response", "skill:web-research",
    }
    assert all(len(d) > 40 for d in INTENT_CHOICES.values())


# ── reading the answer ──────────────────────────────────────────────────────


def test_the_top_choice_is_the_label_however_unsure():
    r = ir.parse(answer("web_search", p_top=0.31))
    assert r.label == "web_search" and r.score == pytest.approx(0.31)
    assert set(r.scores) == set(INTENT_CHOICES)


def test_a_skill_needs_its_own_higher_bar_else_the_best_other_option_wins():
    below = ir.parse(answer("skill:web-research", p_top=ir.MIN_PROB_SKILL - 0.05, web_search=0.2))
    assert below.label == "web_search"
    above = ir.parse(answer("skill:web-research", p_top=ir.MIN_PROB_SKILL + 0.05))
    assert above.label == "skill:web-research"


def test_the_skill_bar_never_applies_to_other_labels():
    assert ir.parse(answer("direct_response", p_top=0.4)).label == "direct_response"
    assert ir.MIN_PROB_SKILL > 0.5 and SKILL_LABEL_PREFIX == "skill:"


def test_missing_probabilities_fall_back_to_the_choice_itself():
    body = {"answers": [{"type": "choice", "name": "intent", "choice": "stock", "confidence": 0.8}]}
    assert ir.parse(body).label == "stock"


@pytest.mark.parametrize("body", [
    {},
    {"answers": []},
    {"answers": [{"type": "refusal", "name": "intent"}]},
    {"answers": [{"type": "choice", "name": "other", "choice": "stock"}]},
    {"answers": [{"type": "choice", "name": "intent", "choice": "made_up"}]},
])
def test_an_answer_that_is_not_one_of_ours_is_a_router_error(body):
    with pytest.raises(ir.RouterError):
        ir.parse(body)


# ── failure ─────────────────────────────────────────────────────────────────


def test_a_non_200_is_a_router_error_not_a_label():
    client = client_for(lambda r: httpx.Response(404, text="no beta access"))
    with pytest.raises(ir.RouterError, match="404"):
        run(ir.route("hi", client=client))


def test_a_missing_key_is_a_router_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ir.RouterError, match="OPENAI_API_KEY"):
        run(ir.route("hi", client=client_for(lambda r: httpx.Response(200, json=answer("weather")))))
