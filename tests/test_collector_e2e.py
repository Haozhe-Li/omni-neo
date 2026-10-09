"""The collector against the real request path, next to POST /chat, byte for byte.

    TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:5432/omni_test \\
    TEST_REDIS_URL=redis://127.0.0.1:6379/15 \\
        venv/bin/python3.12 -m pytest tests/test_collector_e2e.py -q

Opt-in: needs a scratch Postgres (schema.sql is applied) and a plain Redis (no
modules — the LangGraph checkpointer is swapped for an in-memory one). Nothing
else is real-world: the model is a recording stand-in, and the LLM-backed gates
around it (safety check, pre-flight scout, follow-up suggestions) are stubbed.

What it proves is the claim the whole collector rests on — *the model is shown
the same thing it is shown in production*. The same conversation is run twice:

  - through POST /chat, exactly as the frontend calls it (guest id header, the
    personalization object `buildPersonalization` builds, memory read from the
    user's stored `user_memories` row), and
  - through the collector API (key header, explicit fields, memory typed in),

and the messages the model receives — system prompt, tools, every human message —
are compared. Then the collector-only steps are checked: the edit is read by the
next turn, and submit files the right row.
"""

from __future__ import annotations

import os

import pytest

URL = os.getenv("TEST_DATABASE_URL")
REDIS = os.getenv("TEST_REDIS_URL")
pytestmark = pytest.mark.skipif(not (URL and REDIS), reason="TEST_DATABASE_URL / TEST_REDIS_URL not set")

from tests.collector_stack import LUNA, REDIS, URL, stack  # noqa: E402

MEMORY = "## Profile\n- Backend engineer, prefers terse answers\n\n## Current Focus\n- Migrating to Postgres 17"
LOC = "Shanghai, China (IP Approximate)"
DT1 = "2026-10-07T14:05:09+08:00"
DT2 = "2026-10-07T14:09:41+08:00"
GUEST = "guest_e2e-parity"
LUNA = {"model_name": "gpt-6-luna-2026-09-01"}


def test_collector_matches_chat_and_files_an_example():
    with stack() as st:
        client, models, pg = st.client, st.models, st.pg
        human_texts, system_text = st.human_texts, st.system_text
        for t in ("user_memories", "collector_turns"):
            pg.execute(f"DELETE FROM {t}")
        pg.execute("DELETE FROM sft_examples WHERE source = 'collector'")
        pg.execute("DELETE FROM user_usage WHERE user_id = %s", (GUEST,))

        # ── production: POST /chat, twice (turn 1 then turn 3) ──────────────
        pg.execute(
            "INSERT INTO user_memories (user_id, content) VALUES (%s, %s) "
            "ON CONFLICT (user_id) DO UPDATE SET content = EXCLUDED.content", (GUEST, MEMORY),
        )
        guest = {"X-Guest-Id": GUEST}
        prod_thread = client.get("/get_thread_id", headers=guest).json()

        def chat_turn(thread_id: str, query: str, turn: int, dt: str, skill: str | None = None) -> None:
            payload = {
                "query": query, "thread_id": thread_id, "model": "best", "turn": turn,
                "personalization": {
                    "response_language": "zh-CN", "memory_enabled": True,
                    "user_local_datetime": dt, "user_location": LOC,
                },
            }
            if skill:
                payload["skill"] = skill
            with client.stream("POST", "/chat", json=payload, headers=guest) as r:
                assert r.status_code == 200, r.read()
                body = "".join(r.iter_text())
            assert '"type": "done"' in body or '"type":"done"' in body, body[-400:]

        chat_turn(prod_thread, "今天适合跑步吗？", 1, DT1)
        prod_first = human_texts(models["best"])
        chat_turn(prod_thread, "那明天呢？", 3, DT2)
        prod_second = human_texts(models["best"])
        prod_system = system_text(models["best"])

        # ── collector: the same conversation, explicit fields ───────────────
        h = {"X-Collector-Key": "k", "X-Collector-Id": "e2e"}
        thread = client.post("/api/collector/threads", headers=h).json()["thread_id"]

        def collect_turn(query: str, dt: str, memory: str | None, *, thread_id: str | None = None, skill: str | None = None) -> str:
            thread_id = thread_id or thread
            body = {
                "query": query, "thread_id": thread_id, "model": "best",
                "personalization": {"user_local_datetime": dt, "user_location": LOC, "response_language": "zh-CN"},
            }
            if memory:
                body["memory"] = memory
            if skill:
                body["skill"] = skill
            with client.stream("POST", "/api/collector/generate", json=body, headers=h) as r:
                assert r.status_code == 200, r.read()
                text = "".join(r.iter_text())
            assert '"done"' in text, text[-400:]
            state = client.get(f"/api/collector/threads/{thread_id}/state", headers=h).json()
            assert state["complete"], state
            return state["final_text"]

        first_answer = collect_turn("今天适合跑步吗？", DT1, MEMORY)
        col_first = human_texts(models["best"])
        assert first_answer.startswith("answer")

        # an edit lands in the checkpoint and is what the next turn reads
        r = client.put(f"/api/collector/threads/{thread}/final", json={"text": "EDITED ANSWER ONE"}, headers=h)
        assert r.status_code == 200 and r.json()["changed"] is True

        collect_turn("那明天呢？", DT2, None)
        col_second = human_texts(models["best"])
        seen_by_model = [m.content for m in models["best"].calls[-1] if type(m).__name__ == "AIMessage"]
        collected_system = system_text(models["best"])

        # ── the claim ───────────────────────────────────────────────────────
        assert col_first == prod_first, "turn 1: what the model read differs from production"
        assert col_second == prod_second, "turn 3: what the model read differs from production"
        assert collected_system == prod_system
        assert seen_by_model == ["EDITED ANSWER ONE"]
        assert col_first[0].startswith("<user_memory>") and "<system_reminder>" in col_first[0]
        assert "<user_memory>" not in col_second[1]          # memory goes in once, on turn 1

        # ── a skill switched on: same picker id in, same message out ─────────
        for picked, on_disk in (("deep-research", "web-research"), ("guided-learning", "guided-learning")):
            prod_t = client.get("/get_thread_id", headers=guest).json()
            chat_turn(prod_t, "帮我系统研究一下抗生素耐药性", 1, DT1, skill=picked)
            prod_skill = human_texts(models["best"])
            col_t = client.post("/api/collector/threads", headers=h).json()["thread_id"]
            collect_turn("帮我系统研究一下抗生素耐药性", DT1, MEMORY, thread_id=col_t, skill=picked)
            col_skill = human_texts(models["best"])
            assert col_skill == prod_skill, f"skill {picked}: what the model read differs from production"
            assert f"<requested_skill>\n{on_disk}\n</requested_skill>" in col_skill[0]
            assert "<context_enrichment>" in col_skill[0]       # the skill's own instructions, injected once
            turn_rows = client.get(f"/api/collector/threads/{col_t}/state", headers=h).json()["turns"]
            assert turn_rows[0]["skill"] == picked

        # ── submit ──────────────────────────────────────────────────────────
        r = client.post(f"/api/collector/threads/{thread}/submit", json={"note": "e2e"}, headers=h)
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "accepted" and r.json()["edited"] and r.json()["turn"] == 3

        row = pg.fetch_one("SELECT * FROM sft_examples WHERE thread_id = %s", (thread,))
        assert (row["source"], row["status"], row["edited"], row["turn"], row["has_memory"]) == ("collector", "accepted", True, 3, True)
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert [m["content"] for m in row["messages"] if m["role"] == "user"] == col_second
        assert [m["content"] for m in row["messages"] if m["role"] == "assistant"][0] == "EDITED ANSWER ONE"
        assert row["models_seen"] == [LUNA["model_name"]]
        assert row["collect_meta"]["note"] == "e2e"
        assert [t["turn"] for t in row["collect_meta"]["turns"]] == [1, 3]
        assert row["collect_meta"]["turns"][0]["memory"] == MEMORY
        assert row["collect_meta"]["turns"][0]["personalization"]["response_language"] == "zh-CN"

        # an unedited conversation is filed pending, not accepted
        thread2 = client.post("/api/collector/threads", headers=h).json()["thread_id"]
        body = {"query": "hi", "thread_id": thread2, "personalization": {"user_local_datetime": DT1}}
        with client.stream("POST", "/api/collector/generate", json=body, headers=h) as rr:
            "".join(rr.iter_text())
        r = client.post(f"/api/collector/threads/{thread2}/submit", json={}, headers=h)
        assert r.status_code == 200 and r.json()["status"] == "pending" and not r.json()["edited"]

        # another annotator cannot see or touch this thread
        other = {"X-Collector-Key": "k", "X-Collector-Id": "someone-else"}
        assert client.get(f"/api/collector/threads/{thread}/state", headers=other).status_code == 404
        assert client.post(f"/api/collector/threads/{thread}/submit", json={}, headers=other).status_code == 404

        # a thumbs-up on the collector thread is refused: it is not an ordinary chat thread
        # (and the thread belongs to a collector id, not a Clerk/guest user)
        r = client.post(f"/api/threads/{thread}/feedback", json={"turn": 3, "rating": "up"}, headers=guest)
        assert r.status_code in (403, 404)

        pg.execute("DELETE FROM sft_examples WHERE source = 'collector'")
        pg.execute("DELETE FROM user_memories WHERE user_id = %s", (GUEST,))
        pg.execute("DELETE FROM collector_turns")

