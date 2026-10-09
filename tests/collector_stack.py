"""Shared stack for the opt-in end-to-end collector tests (needs Postgres + Redis).

`stack()` builds the real routers (chat, collector, threads) on real Redis and Postgres
with a recording stand-in for the model, and stubs only the LLM-backed gates around it
(safety check, pre-flight scout, follow-ups) and the harness capture. Not a test module.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from types import SimpleNamespace

URL = os.getenv("TEST_DATABASE_URL")
REDIS = os.getenv("TEST_REDIS_URL")
LUNA = {"model_name": "gpt-6-luna-2026-09-01"}


def fresh_pools() -> None:
    """Drop pools a previous test left bound to a loop that has since closed."""
    from core.database import pg

    pg.close_pools()
    pg._async_pool = None
    pg._async_lock = None
    # The async Redis clients are cached module globals too, bound to the loop that made them.
    import core.utils.redis_client as rc

    rc._async_client = None
    rc._blocking_client = None


@contextmanager
def stack():
    from tests.test_collector_parity import _IMPORT_ENV

    for k, v in _IMPORT_ENV.items():
        os.environ.setdefault(k, v)
    os.environ["DATABASE_URL"] = URL
    os.environ["REDIS_URL"] = REDIS
    os.environ["COLLECTOR_API_KEY"] = "k"
    fresh_pools()
    import psycopg

    with psycopg.connect(URL, autocommit=True) as conn:
        conn.execute(open(os.path.join(os.path.dirname(__file__), "..", "schema.sql")).read())

    from deepagents import create_deep_agent
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langgraph.checkpoint.memory import InMemorySaver
    from pydantic import Field

    import core.agent as agent_mod
    import core.harness_snapshot as harness
    import core.routers.chat as chat_mod
    import core.stream as stream_mod
    from core.context_enrichment import Enrichment
    from core.database import pg
    from core.routers import chat, collector

    class Recorder(BaseChatModel):
        calls: list = Field(default_factory=list)

        @property
        def _llm_type(self) -> str:
            return "recorder"

        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            self.calls.append(list(messages))
            n = len(self.calls)
            msg = AIMessage(content=f"answer {n}", response_metadata=dict(LUNA))
            return ChatResult(generations=[ChatGeneration(message=msg)])

        def _stream(self, messages, stop=None, run_manager=None, **kw):
            # The page streams tokens, so the stand-in must too (stream_mode="messages").
            self.calls.append(list(messages))
            n = len(self.calls)
            for piece in (f"answer ", f"{n}"):
                yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
            yield ChatGenerationChunk(message=AIMessageChunk(content="", response_metadata=dict(LUNA), chunk_position="last"))

    saver = InMemorySaver()
    models: dict[str, Recorder] = {}
    agent_mod._agents.clear()
    for model_id in agent_mod.CHAT_MODELS:
        models[model_id] = Recorder()
        agent_mod._agents[model_id] = create_deep_agent(
            model=models[model_id], tools=[], checkpointer=saver,
            system_prompt="SYSTEM PROMPT UNDER TEST", skills=None,
        )

    async def not_harmful(_q):
        return False

    async def no_enrichment(*a, **kw):
        return Enrichment()

    async def no_follow_ups(*a, **kw):
        return []

    async def fake_harness():
        return ("e" * 32, "SYSTEM PROMPT UNDER TEST", [])

    stream_mod.is_harmful = not_harmful
    stream_mod.enrich_context = no_enrichment
    chat_mod.get_follow_ups = no_follow_ups
    collector.live_harness = fake_harness
    chat_mod.PERSIST_GRACE_SECONDS = 0
    import core.routers.state as state_mod
    state_mod.PERSIST_GRACE_SECONDS = 0

    app = FastAPI()
    app.include_router(chat.router)
    app.include_router(collector.router)
    app.include_router(__import__("core.routers.threads", fromlist=["router"]).router)
    app.include_router(__import__("core.routers.uploads", fromlist=["router"]).router)

    def human_texts(model: Recorder) -> list[str]:
        """The user messages the model was last shown, in order."""
        last = model.calls[-1]
        return [m.content for m in last if type(m).__name__ == "HumanMessage"]

    def system_text(model: Recorder) -> str:
        def flat(c):
            return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))

        return "".join(flat(m.content) for m in model.calls[-1] if type(m).__name__ == "SystemMessage")


    from fastapi.testclient import TestClient

    ns = SimpleNamespace(models=models, human_texts=human_texts, system_text=system_text, pg=pg,
                         stream_mod=stream_mod, collector=collector)
    with TestClient(app) as client:
        ns.client = client
        yield ns
    fresh_pools()  # the async pool belongs to the client's loop and dies with it
