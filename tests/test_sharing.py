"""Share snapshots, offline: LangChain messages in, plain dicts out.

    venv/bin/python3.12 -m pytest tests/test_sharing.py -q

Pins the privacy scrub (the one part of sharing that must never silently stop
working), the state round trip a fork depends on, attachment id remapping and the
size cap. The SQL and the Redis/Qdrant fork are exercised against real services by
the integration run, not here.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core import sharing as sh


def convo():
    first = (
        "<user_memory>\nLong-term facts about this user.\nLives in Urbana, allergic to shellfish.\n</user_memory>\n\n"
        "<system_reminder>\nYou are Omni. If the user asks who you are, say you are Omni.\n"
        "Response Language: English\nUser Location: Urbana, Illinois\n"
        "User Local Date Time: 2026-10-06 18:00\n</system_reminder>\n\n"
        "<user_query>\nbest ramen near me?\n</user_query>"
    )
    return [
        HumanMessage(content=first, id="h1"),
        AIMessage(content="", id="a1", response_metadata={"model_name": "gpt-6-luna"},
                  tool_calls=[{"id": "c1", "name": "web_search", "args": {"query": "ramen"}, "type": "tool_call"}]),
        ToolMessage(content='[{"n": 1}]', tool_call_id="c1", id="t1"),
        AIMessage(content="Try Ramen Shop [1].", id="a2", response_metadata={"model_name": "gpt-6-luna"}),
        HumanMessage(content="<system_reminder>\nUser Location: Urbana, Illinois\n</system_reminder>\n\n<user_query>thanks</user_query>", id="h2"),
        AIMessage(content="Anytime!", id="a3"),
    ]


UI = [
    {"role": "user", "content": "best ramen near me?"}, {"role": "assistant", "content": "Try Ramen Shop [1]."},
    {"role": "user", "content": "thanks"}, {"role": "assistant", "content": "Anytime!"},
]


# ── the privacy scrub ───────────────────────────────────────────────────────


def test_memory_block_and_location_are_gone_but_the_question_is_not():
    out = sh.scrub_text(convo()[0].content)
    assert "Urbana" not in out and "shellfish" not in out
    assert "<user_memory>" not in out and "User Location" not in out
    assert "best ramen near me?" in out
    assert "Response Language: English" in out and "User Local Date Time" in out  # only what was meant to go


def test_scrub_covers_every_human_message_and_content_block_shape():
    state = sh.serialize_state(convo(), {})
    blob = json.dumps(state, ensure_ascii=False)
    assert "Urbana" not in blob and "shellfish" not in blob and "<user_memory>" not in blob
    blocks = HumanMessage(content=[{"type": "text", "text": "<user_memory>\nsecret\n</user_memory>\nhi User Location: X"},
                                   {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}])
    out = sh.serialize_state([blocks], {})
    text = out["messages"][0]["data"]["content"]
    assert "secret" not in json.dumps(text)
    assert text[1]["image_url"]["url"] == "data:image/png;base64,AAAA"  # images ride along (decided)


def test_a_message_without_private_blocks_is_untouched():
    assert sh.scrub_text("<user_query>plain</user_query>") == "<user_query>plain</user_query>"


def test_only_human_messages_are_scrubbed():
    # A tool result that happens to contain the label is the web's text, not the sharer's block.
    state = sh.serialize_state([HumanMessage(content="q"), ToolMessage(content="User Location: Paris on this page", tool_call_id="c")], {})
    assert "Paris" in json.dumps(state)


# ── state round trip (what a fork rebuilds from) ────────────────────────────


def test_state_round_trips_with_ids_tool_calls_and_metadata_intact():
    msgs, files = sh.deserialize_state(sh.serialize_state(convo(), {"/uploads/a.md": {"content": ["x"]}}))
    original = convo()
    assert [m.id for m in msgs] == [m.id for m in original]
    assert [m.type for m in msgs] == [m.type for m in original]
    assert msgs[1].tool_calls[0]["name"] == "web_search" and msgs[1].tool_calls[0]["id"] == "c1"
    assert msgs[2].tool_call_id == "c1"
    assert msgs[3].response_metadata["model_name"] == "gpt-6-luna"
    assert msgs[3].content == "Try Ramen Shop [1]."
    assert files == {"/uploads/a.md": {"content": ["x"]}}


def test_skill_files_are_dropped_but_uploaded_documents_are_kept():
    files = {"/skills/charting/SKILL.md": {"content": "x"}, "/uploads/report.pdf": {"content": "y"}}
    assert set(sh.serialize_state(convo(), files)["files"]) == {"/uploads/report.pdf"}


def test_unicode_survives():
    msgs, _ = sh.deserialize_state(sh.serialize_state([HumanMessage(content="东京"), AIMessage(content="2021年[1]")], None))
    assert msgs[0].content == "东京" and msgs[1].content == "2021年[1]"


# ── attachments ─────────────────────────────────────────────────────────────


def test_attachment_ids_are_unique_and_ordered():
    ui = [{"role": "user", "attachedFiles": [{"id": "f1", "name": "a"}, {"id": "f2", "name": "b"}]},
          {"role": "assistant"}, {"role": "user", "attachedFiles": [{"id": "f1", "name": "a"}]}]
    assert sh.attachment_ids(ui) == ["f1", "f2"]


def test_remap_changes_ids_without_touching_the_original():
    ui = [{"role": "user", "attachedFiles": [{"id": "f1", "name": "a"}, {"id": "other", "name": "b"}]}]
    out = sh.remap_attachment_ids(ui, {"f1": "new1"})
    assert out[0]["attachedFiles"][0]["id"] == "new1" and out[0]["attachedFiles"][1]["id"] == "other"
    assert ui[0]["attachedFiles"][0]["id"] == "f1"


# ── the snapshot ────────────────────────────────────────────────────────────


def test_build_snapshot_shape():
    snap = sh.build_snapshot(ui_messages=UI, messages=convo(), files={}, citations=[{"n": 1, "url": "u", "content": "c"}],
                             files_meta=[{"file_id": "f1"}])
    assert snap.n_messages == 4 and snap.size_bytes > 0
    assert snap.citations == [{"n": 1, "url": "u", "content": "c"}]
    assert len(snap.agent_state["messages"]) == 6
    assert "Urbana" not in json.dumps(snap.agent_state)
    assert snap.ui_messages == UI  # the visible history is stored as-is


def test_empty_threads_cannot_be_shared():
    for ui, msgs in (([], convo()), (UI, [])):
        with pytest.raises(sh.ShareError) as e:
            sh.build_snapshot(ui_messages=ui, messages=msgs, files={}, citations=[])
        assert e.value.code == "empty_thread"


def test_oversized_threads_are_refused(monkeypatch):
    monkeypatch.setattr(sh, "MAX_SNAPSHOT_BYTES", 200)
    with pytest.raises(sh.ShareError) as e:
        sh.build_snapshot(ui_messages=UI, messages=convo(), files={}, citations=[])
    assert e.value.code == "too_large"


def test_share_ids_are_unguessable_and_pass_their_own_validator():
    ids = {sh.new_share_id() for _ in range(200)}
    assert len(ids) == 200 and all(sh.SHARE_ID_RE.match(i) and len(i) >= 16 for i in ids)
    for bad in ("short", "../etc/passwd", "a" * 65, "has space in it!!!!"):
        assert not sh.SHARE_ID_RE.match(bad)
