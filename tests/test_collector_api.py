"""POST /api/collector/generate at the HTTP seam, with the agent stubbed out.

    venv/bin/python3.12 -m pytest tests/test_collector_api.py -q

The parity tests pin the *functions* the collector shares with /chat. This pins
the *call*: what `_generate_background` — the one function that actually runs the
agent, shared with /chat — is handed when a collector request arrives. Everything
around it (Postgres, Redis, the checkpoint) is replaced; the assertions are on
the arguments, because those are the model's whole input.
"""

from __future__ import annotations

import os

import pytest

from tests.test_collector_parity import _IMPORT_ENV, DT, LOC, MEMORY

for _k, _v in _IMPORT_ENV.items():
    os.environ.setdefault(_k, _v)
os.environ["COLLECTOR_API_KEY"] = "k"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from core.routers import collector  # noqa: E402

HEADERS = {"X-Collector-Key": "k", "X-Collector-Id": "alice"}
OWNER = "collector_alice"


def body(**over):
    base = {
        "query": "今天适合跑步吗？",
        "thread_id": "t-1",
        "personalization": {"user_local_datetime": DT, "user_location": LOC, "response_language": "zh-CN"},
        "model": "best",
        "memory": MEMORY,
    }
    base.update(over)
    return base


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(collector.router)
    return TestClient(app)


@pytest.fixture
def wired(monkeypatch):
    """Stub every collaborator of `generate`; return the dict the stubs record into."""
    seen: dict = {"kwargs": None, "recorded": [], "history": []}

    async def owned(thread_id, user_id):
        assert user_id == OWNER
        return {"origin": "collector", "is_locked": False}

    async def messages(thread_id):
        return list(seen["history"])

    async def not_generating(thread_id):
        return False

    async def alist_turns(thread_id, user_id):
        return [{"turn": t} for t in seen.get("recorded_turns", [])]

    async def arecord_turn(**kw):
        seen["recorded"].append(kw)

    async def stream_begin(thread_id):
        return None

    async def generate_background(**kw):
        seen["kwargs"] = kw

    async def stream_read(thread_id):
        yield "data: {\"type\": \"done\"}\n\n"

    monkeypatch.setattr(collector, "_owned_thread", owned)
    monkeypatch.setattr(collector, "_checkpoint_messages", messages)
    monkeypatch.setattr(collector, "stream_is_generating", not_generating)
    monkeypatch.setattr(collector.db_collector, "alist_turns", alist_turns)
    monkeypatch.setattr(collector.db_collector, "arecord_turn", arecord_turn)
    monkeypatch.setattr(collector, "stream_begin", stream_begin)
    monkeypatch.setattr(collector, "_generate_background", generate_background)
    monkeypatch.setattr(collector, "stream_read", stream_read)
    return seen


def exchange(n: int):
    return [HumanMessage(content=f"q{n}"), AIMessage(content=f"a{n}", response_metadata={"model_name": "gpt-6-luna"})]


class TestAuth:
    def test_no_key(self, client):
        assert client.post("/api/collector/generate", json=body()).status_code == 401

    def test_wrong_key(self, client):
        r = client.post("/api/collector/generate", json=body(), headers={**HEADERS, "X-Collector-Key": "no"})
        assert r.status_code == 401

    def test_disabled(self, client, monkeypatch):
        monkeypatch.delenv("COLLECTOR_API_KEY")
        assert client.post("/api/collector/generate", json=body(), headers=HEADERS).status_code == 503

    def test_every_route_is_guarded(self, client):
        for method, path in [
            ("post", "/api/collector/threads"),
            ("delete", "/api/collector/threads/x"),
            ("post", "/api/collector/generate"),
            ("get", "/api/collector/threads/x/stream"),
            ("post", "/api/collector/threads/x/stop"),
            ("get", "/api/collector/threads/x/state"),
            ("put", "/api/collector/threads/x/final"),
            ("post", "/api/collector/threads/x/submit"),
        ]:
            assert getattr(client, method)(path).status_code in (401, 422), (method, path)


class TestGenerateInput:
    def test_first_turn_is_handed_over_exactly_as_chat_would(self, client, wired):
        r = client.post("/api/collector/generate", json=body(), headers=HEADERS)
        assert r.status_code == 200
        kw = wired["kwargs"]
        assert kw["turn"] == 1
        assert kw["query"] == "今天适合跑步吗？"
        assert kw["model_id"] == "best"
        assert kw["user_location"] == LOC and kw["user_local_datetime"] == DT
        assert kw["system_reminder"] == (
            "You are Omni. If the user asks who you are, say you are Omni.\n"
            "Response Language: zh-CN\n"
            f"User Location: {LOC}\n"
            f"User Local Date Time: {DT}\n"
        )
        assert kw["user_memory"].endswith(MEMORY) and kw["user_memory"].startswith("Long-term facts")
        # the things the collector must never switch on
        assert kw["memory_enabled"] is False
        assert (kw["attached_file_ids"], kw["skill"], kw["source_url"], kw["follow_up_content"]) == (None,) * 4

    def test_the_turn_comes_from_the_checkpoint_not_the_client(self, client, wired):
        wired["history"] = exchange(1) + exchange(2)
        wired["recorded_turns"] = [1, 3]
        r = client.post("/api/collector/generate", json=body(memory=None), headers=HEADERS)
        assert r.status_code == 200
        assert wired["kwargs"]["turn"] == 5
        assert wired["recorded"][0]["turn"] == 5
        assert wired["kwargs"]["user_memory"] == ""

    def test_the_client_cannot_send_a_turn(self, client, wired):
        assert client.post("/api/collector/generate", json=body(turn=1), headers=HEADERS).status_code == 422

    def test_memory_after_the_first_turn_is_refused(self, client, wired):
        wired["history"] = exchange(1)
        wired["recorded_turns"] = [1]
        r = client.post("/api/collector/generate", json=body(), headers=HEADERS)
        assert r.status_code == 400 and wired["kwargs"] is None

    def test_a_turn_that_never_finished_blocks_the_thread(self, client, wired):
        wired["history"] = [HumanMessage(content="q1")]
        wired["recorded_turns"] = [1]
        assert client.post("/api/collector/generate", json=body(memory=None), headers=HEADERS).status_code == 409
        assert wired["kwargs"] is None

    def test_a_turn_that_bypassed_the_collector_blocks_the_thread(self, client, wired):
        wired["history"] = exchange(1)
        wired["recorded_turns"] = []
        assert client.post("/api/collector/generate", json=body(memory=None), headers=HEADERS).status_code == 409

    def test_an_attempt_that_never_reached_the_checkpoint_is_retried(self, client, wired):
        wired["recorded_turns"] = [1]  # recorded, then refused before the model ran
        assert client.post("/api/collector/generate", json=body(), headers=HEADERS).status_code == 200
        assert wired["kwargs"]["turn"] == 1

    def test_oversized_memory_is_refused(self, client, wired):
        r = client.post("/api/collector/generate", json=body(memory="x" * 3001), headers=HEADERS)
        assert r.status_code == 422 and wired["kwargs"] is None

    @pytest.mark.parametrize(
        "bad",
        [
            {"model": "rix"},
            {"personalization": {"user_local_datetime": "2026-10-07T14:05:09Z"}},
            {"personalization": {"user_local_datetime": DT, "response_language": "auto"}},
            {"skill": "web-research"},
        ],
    )
    def test_non_production_inputs_never_reach_the_agent(self, client, wired, bad):
        r = client.post("/api/collector/generate", json=body(**bad), headers=HEADERS)
        assert r.status_code == 422 and wired["kwargs"] is None
