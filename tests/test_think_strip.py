"""Gemma's empty thought block must never reach the user, the callbacks or the thread."""
import asyncio
import os

os.environ.setdefault("WANDB_API_KEY", "test")

import pytest
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_openai import ChatOpenAI

from core.llm import ChatWandbGemma
from core.think_strip import ThinkBlockStripper


def _run(pieces):
    s = ThinkBlockStripper()
    return "".join(s.feed(p) for p in pieces) + s.flush()


@pytest.mark.parametrize("pieces", [
    ["<|channel>thought\n<channel|>Hello"],
    ["<|chan", "nel>thought\n<chan", "nel|>Hel", "lo"],
    ["thought\n<channel|>Hello"],
    [":thought\nHello"],
    ["Hello"],
])
def test_marker_is_removed_however_it_is_split(pieces):
    assert _run(pieces) == "Hello"


def test_the_word_thought_inside_an_answer_survives():
    assert _run(["I thought\nabout it"]) == "I thought\nabout it"


def test_chat_model_filters_before_callbacks_fire(monkeypatch):
    pieces = ["<|channel>thou", "ght\n<channel|>", "Hi ", "there"]

    async def fake(self, messages, stop=None, run_manager=None, **kw):
        assert run_manager is None  # the base class must not fire callbacks itself
        for p in pieces:
            yield ChatGenerationChunk(message=AIMessageChunk(content=p))

    monkeypatch.setattr(ChatOpenAI, "_astream", fake)

    class Mgr:
        tokens = []

        async def on_llm_new_token(self, token, chunk=None, **kw):
            self.tokens.append(token)

    llm = ChatWandbGemma(model="x")

    async def go():
        return [g.message.content async for g in llm._astream([], run_manager=Mgr())]

    out = asyncio.run(go())
    assert "".join(out) == "Hi there"
    assert "".join(Mgr.tokens) == "Hi there"
