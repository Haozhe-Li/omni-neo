"""The chat model catalog and its fallbacks, offline.

    venv/bin/python3.12 -m pytest tests/test_chat_models.py -q
"""

from __future__ import annotations

import os

import dotenv
import pytest

# core.llm builds every provider client at import time, which wants a key for
# each. Real ones from .env when present; placeholders otherwise — nothing here
# makes a request.
dotenv.load_dotenv()
for _key in ("OPENAI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "CEREBRAS_API_KEY", "WANDB_API_KEY"):
    os.environ.setdefault(_key, "test-placeholder")

from core import llm  # noqa: E402
from core.chat_models import CHAT_MODELS, credits_for, resolve_model, serving_model  # noqa: E402


def test_the_archived_gemma_is_gone_everywhere_it_could_be_served_from():
    # Cerebras archived gemma-4-31b (404 model_archived_error).
    assert "gemma" not in CHAT_MODELS
    assert not hasattr(llm, "gemma_4_31b") and not hasattr(llm, "gemma_4_31b_high")
    assert all(getattr(m, "model_name", "") != "gemma-4-31b" for m in llm.CHAT_LLM_FALLBACKS)
    assert llm.CHAT_LLM_FALLBACKS, "the chain must keep at least one live fallback"


def test_old_threads_that_carry_gemma_still_resolve_and_bill_by_route():
    # A rewind of a thread created while gemma was selectable sends `gemma`; an
    # unknown id would 400 it. It resolves to `best`, which is billed as the
    # model it routes to.
    assert resolve_model("gemma").id == "best"
    assert resolve_model("GEMMA ").id == "best"
    assert credits_for("gemma") == credits_for("best") == 3.0


def test_best_is_a_router_and_is_billed_as_the_model_it_routes_to():
    best = CHAT_MODELS["best"]
    assert best.credits is None  # no price of its own
    # Both routes name real, concrete (non-router) entries.
    for route in (best.text_route, best.vision_route):
        assert route in CHAT_MODELS and CHAT_MODELS[route].text_route is None
    # Today both routes are luna: 3 credits on text and on images.
    assert serving_model("best").id == "luna"
    assert serving_model("best", has_image=True).id == "luna"
    assert credits_for("best") == credits_for("best", has_image=True) == 3.0
    # The route must be the model that actually runs, or billing and serving diverge.
    assert best.llm is CHAT_MODELS[best.text_route].llm
    assert best.vision_fallback is CHAT_MODELS[best.vision_route].llm


def test_concrete_models_bill_for_themselves():
    assert (credits_for("rix"), credits_for("luna"), credits_for("gemini")) == (1.0, 3.0, 3.0)
    assert credits_for("rix", has_image=True) == 1.0
    assert all(serving_model(k).id == k for k in ("rix", "luna", "gemini"))


def test_every_billing_key_has_a_price():
    from core.database.db_user_usage import MODE_CREDIT_COST

    # What the chat router charges for any model, with or without an image.
    keys = {serving_model(k, has_image=i).id for k in CHAT_MODELS for i in (False, True)}
    assert keys <= set(MODE_CREDIT_COST)
    # Router and legacy keys are gone: nothing is billed under them any more.
    assert not {"best", "best-vision", "fast", "pro"} & set(MODE_CREDIT_COST)
    assert MODE_CREDIT_COST["scheduled"] == 3.0
    # The table agrees with the catalog.
    for k in ("rix", "luna", "gemini"):
        assert MODE_CREDIT_COST[k] == CHAT_MODELS[k].credits


def test_the_remaining_models_are_unchanged():
    assert set(CHAT_MODELS) == {"best", "rix", "luna", "gemini"}
    assert [resolve_model(k).id for k in ("fast", "pro", None)] == ["best", "best", "best"]


def test_unknown_ids_still_fail_loudly():
    with pytest.raises(ValueError):
        resolve_model("gpt-9000")


def test_credit_limits_are_fixed_constants():
    from core.database import db_user_usage as u

    assert (u.GUEST_DAILY_CREDIT_LIMIT, u.GUEST_MONTHLY_CREDIT_LIMIT) == (100, 300)
    assert (u.USER_DAILY_CREDIT_LIMIT, u.USER_MONTHLY_CREDIT_LIMIT) == (200, 5000)
    assert u._limits("guest_abc") == (100, 300)
    assert u._limits("user_abc") == (200, 5000)
