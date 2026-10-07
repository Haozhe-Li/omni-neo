"""The collector's SQL against a real Postgres.

    TEST_DATABASE_URL=postgresql://postgres@127.0.0.1:5432/omni_test \\
        venv/bin/python3.12 -m pytest tests/test_collector_db.py -q

Skipped unless TEST_DATABASE_URL points at a database to write into (it applies
schema.sql itself, twice, to prove the migration is idempotent on a database that
already holds thumbs-up rows). Use a scratch database: the test creates and
deletes its own rows by thread id, but it does apply the schema.

Covers what the offline tests cannot: `asave_collector_example` overwriting on
re-submit, thumbs-up rows keeping their defaults beside collector rows, the edit
bookkeeping in collector_turns, and the builder's SELECT still working.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import pathlib
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage

URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TEST_DATABASE_URL is not set")

ROOT = pathlib.Path(__file__).resolve().parents[1]
LUNA = {"model_name": "gpt-6-luna"}
TOOLS = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}]


def _captured(answer: str):
    from core import sft_capture as sc

    msgs = [
        HumanMessage(content="<user_memory>\nm\n</user_memory>\n\n<user_query>\nq\n</user_query>"),
        AIMessage(content=answer, response_metadata=dict(LUNA)),
    ]
    return sc.build_capture(msgs, 1, allow_memory=True)


def test_collector_rows_end_to_end():
    os.environ["DATABASE_URL"] = URL
    import psycopg

    with psycopg.connect(URL, autocommit=True) as conn:
        for _ in range(2):  # second pass: the migration must be a no-op
            conn.execute((ROOT / "schema.sql").read_text())

    from core.database import db_collector, db_sft_examples as sft, pg

    async def go():
        try:
            await pg.aexecute("DELETE FROM sft_examples WHERE thread_id IN ('c-thumb', 'c-col')")
            await pg.aexecute("DELETE FROM collector_turns WHERE thread_id = 'c-col'")
            common = dict(harness_hash="h" * 32, system_prompt="SYS", tools=TOOLS, deepagents_version="0")

            # a thumbs-up row is untouched by the new columns
            await sft.asave_example(thread_id="c-thumb", turn=1, user_id="u1", cap=_captured("a"), **common)
            row = await pg.afetch_one("SELECT source, edited, original_final_text, collect_meta FROM sft_examples WHERE thread_id='c-thumb'")
            assert row == {"source": "thumbs", "edited": False, "original_final_text": None, "collect_meta": None}

            # turns: record, edit twice (original is kept from the first edit), re-record drops the edit
            await db_collector.arecord_turn(
                thread_id="c-col", turn=1, user_id="collector_a", model="best",
                personalization={"user_local_datetime": "2026-10-07T14:05:09+08:00"}, memory="mem",
            )
            assert await db_collector.arecord_edit("c-col", 1, "collector_a", "orig", "v1")
            assert await db_collector.arecord_edit("c-col", 1, "collector_a", "v1", "v2")
            [t] = await db_collector.alist_turns("c-col", "collector_a")
            assert (t["original_final_text"], t["edited_final_text"], t["memory"]) == ("orig", "v2", "mem")
            assert not await db_collector.arecord_edit("c-col", 1, "someone_else", "x", "y")  # scoped to owner
            assert await db_collector.alist_turns("c-col", "collector_b") == []

            # submit, then submit again with a different outcome: the row is replaced
            meta = {"note": "n", "turns": [{"turn": 1}]}
            first = await sft.asave_collector_example(
                thread_id="c-col", turn=1, user_id="collector_a", cap=_captured("v2"),
                status="accepted", edited=True, original_final_text="orig", collect_meta=meta, **common,
            )
            second = await sft.asave_collector_example(
                thread_id="c-col", turn=1, user_id="collector_a", cap=_captured("v3"),
                status="pending", edited=False, original_final_text=None, collect_meta={"note": "again"}, **common,
            )
            assert first == second
            rows = await pg.afetch_all("SELECT * FROM sft_examples WHERE thread_id='c-col'")
            assert len(rows) == 1
            r = rows[0]
            assert (r["source"], r["status"], r["edited"], r["has_memory"]) == ("collector", "pending", False, True)
            assert r["collect_meta"] == {"note": "again"}
            assert r["messages"][-1]["content"] == "v3"

            await db_collector.adelete_turns("c-col", "collector_a")
            assert await db_collector.alist_turns("c-col", "collector_a") == []
        finally:
            await pg.aexecute("DELETE FROM sft_examples WHERE thread_id IN ('c-thumb', 'c-col')")
            await pg.aexecute("DELETE FROM collector_turns WHERE thread_id = 'c-col'")
            await pg.aclose_pools()

    asyncio.run(go())

    # the dataset builder's own query, with the new column, and its memory filter
    spec = importlib.util.spec_from_file_location("rix_build_db", ROOT / "finetune" / "rix_gemma" / "build_dataset.py")
    build = importlib.util.module_from_spec(spec)
    sys.modules["rix_build_db"] = build
    spec.loader.exec_module(build)
    rows = build.fetch_examples(["accepted", "pending"])
    assert all("source" in r for r in rows)
    pg.close_pools()
