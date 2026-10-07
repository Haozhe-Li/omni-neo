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
from core.chat_models import CHAT_MODELS, credits_for, resolve_model  # noqa: E402


def test_the_archived_gemma_is_gone_everywhere_it_could_be_served_from():
    # Cerebras archived gemma-4-31b (404 model_archived_error).
    assert "gemma" not in CHAT_MODELS
    assert not hasattr(llm, "gemma_4_31b") and not hasattr(llm, "gemma_4_31b_high")
    assert all(getattr(m, "model_name", "") != "gemma-4-31b" for m in llm.CHAT_LLM_FALLBACKS)
    assert llm.CHAT_LLM_FALLBACKS, "the chain must keep at least one live fallback"


def test_old_threads_that_carry_gemma_still_resolve_and_bill_as_best():
    # A rewind of a thread created while gemma was selectable sends `gemma`; an
    # unknown id would 400 it.
    assert resolve_model("gemma").id == "best"
    assert resolve_model("GEMMA ").id == "best"
    assert credits_for("gemma") == credits_for("best") == 1.0


def test_the_remaining_models_are_unchanged():
    assert set(CHAT_MODELS) == {"best", "rix", "luna", "gemini"}
    assert [resolve_model(k).id for k in ("fast", "pro", None)] == ["best", "best", "best"]


def test_unknown_ids_still_fail_loudly():
    with pytest.raises(ValueError):
        resolve_model("gpt-9000")
