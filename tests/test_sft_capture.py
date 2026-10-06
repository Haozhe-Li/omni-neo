"""Thumbs-up -> training example conversion, offline: LangChain messages in,
OpenAI-style dicts out. No Redis, no Postgres, no agent.

    venv/bin/python3.12 -m pytest tests/test_sft_capture.py -q

What this pins: turn slicing, the tool-call/result pairing guarantee, image
extraction, the privacy flags, teacher gating, and the UI/checkpoint alignment
check. It says nothing about the SQL (schema.sql + core/database/db_sft_examples.py
need a real Postgres) or about whether luna's `response_metadata` carries the
model name in production — see test_model_name_comes_from_response_metadata for
the assumption this relies on.
"""

from __future__ import annotations

import base64
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core import sft_capture as sc

LUNA = {"model_name": "gpt-6-luna-2026-09-01"}
OTHER = {"model_name": "gemini-3.6-flash"}


def human(text: str, **kw) -> HumanMessage:
    return HumanMessage(content=text, **kw)


def ai(text: str = "", calls: list[tuple[str, str, dict]] | None = None, meta=LUNA) -> AIMessage:
    return AIMessage(
        content=text,
        tool_calls=[{"id": i, "name": n, "args": a, "type": "tool_call"} for i, n, a in (calls or [])],
        response_metadata=dict(meta),
    )


def tool(call_id: str, text: str) -> ToolMessage:
    return ToolMessage(content=text, tool_call_id=call_id)


def one_search_turn(q: str = "who won?", a: str = "Spain [1].") -> list:
    return [
        human(f"<user_query>\n{q}\n</user_query>"),
        ai(calls=[("c1", "web_search", {"query": q})]),
        tool("c1", '[{"n": 1, "title": "t"}]'),
        ai(a),
    ]


# ── turn slicing ────────────────────────────────────────────────────────────


def test_slice_stops_before_the_next_user_turn():
    msgs = one_search_turn("first") + one_search_turn("second")
    first = sc.slice_through_turn(msgs, 1)
    assert first == msgs[:4]
    assert sc.slice_through_turn(msgs, 3) == msgs  # last turn: through the end


def test_even_or_missing_turns_are_rejected():
    msgs = one_search_turn()
    for bad in (0, 2, -1):
        with pytest.raises(sc.CaptureError) as e:
            sc.slice_through_turn(msgs, bad)
        assert e.value.code == "bad_turn"
    with pytest.raises(sc.CaptureError) as e:
        sc.slice_through_turn(msgs, 3)
    assert e.value.code == "turn_not_found"


def test_a_turn_still_in_flight_is_incomplete():
    # ends on a tool result, not an answer
    with pytest.raises(sc.CaptureError) as e:
        sc.slice_through_turn(one_search_turn()[:3], 1)
    assert e.value.code == "turn_incomplete"
    # ends on an assistant message that is only a tool call
    with pytest.raises(sc.CaptureError) as e:
        sc.slice_through_turn(one_search_turn()[:2], 1)
    assert e.value.code == "turn_incomplete"


def test_thumbing_an_earlier_turn_ignores_what_came_after():
    msgs = one_search_turn("first") + [human("second"), ai(calls=[("c9", "web_search", {})])]  # 2nd unfinished
    cap = sc.build_capture(msgs, 1)
    assert cap.messages[-1]["content"] == "Spain [1]."
    assert len(cap.messages) == 4


# ── conversion ──────────────────────────────────────────────────────────────


def test_messages_are_openai_shaped_and_match_the_existing_trace_format():
    cap = sc.build_capture(one_search_turn(), 1)
    roles = [m["role"] for m in cap.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    call = cap.messages[1]["tool_calls"][0]
    # Same shape finetune/pro_agent/collect.py writes: arguments is a JSON *string*.
    assert call == {
        "id": "c1",
        "type": "function",
        "function": {"name": "web_search", "arguments": json.dumps({"query": "who won?"})},
    }
    assert cap.messages[2] == {"role": "tool", "tool_call_id": "c1", "content": '[{"n": 1, "title": "t"}]'}
    assert (cap.n_assistant_turns, cap.n_tool_calls, cap.tools_used) == (2, 1, ["web_search"])
    assert not cap.messages[1]["content"]  # a pure tool-call turn has empty text


def test_unicode_survives_untouched():
    msgs = [human("<user_query>东京奥运会</user_query>"), ai("2021年举行[1]。")]
    cap = sc.build_capture(msgs, 1)
    assert cap.messages[0]["content"] == "<user_query>东京奥运会</user_query>"
    assert cap.messages[1]["content"] == "2021年举行[1]。"


def test_content_blocks_flatten_to_text():
    block_ai = AIMessage(
        content=[
            {"type": "reasoning", "summary": []},
            {"type": "text", "text": "Hello ", "annotations": []},
            {"type": "text", "text": "world"},
        ],
        response_metadata=LUNA,
    )
    cap = sc.build_capture([human("hi"), block_ai], 1)
    assert cap.messages[1]["content"] == "Hello world"


def test_reasoning_only_message_is_dropped_not_trained_as_silence():
    msgs = [human("hi"), AIMessage(content=[{"type": "reasoning", "summary": []}], response_metadata=LUNA), ai("Hi!")]
    cap = sc.build_capture(msgs, 1)
    assert [m["role"] for m in cap.messages] == ["user", "assistant"]
    assert cap.messages[1]["content"] == "Hi!"


def test_unpaired_tool_calls_are_refused():
    msgs = [
        human("q"),
        ai(calls=[("a", "web_search", {}), ("b", "fetch_url", {})]),
        tool("a", "r"),  # b never answered (interrupted)
        ai("done"),
    ]
    with pytest.raises(sc.CaptureError) as e:
        sc.build_capture(msgs, 1)
    assert e.value.code == "unpaired_tool_calls"


def test_parallel_tool_calls_pair_up():
    msgs = [
        human("q"),
        ai(calls=[("a", "web_search", {}), ("b", "fetch_url", {})]),
        tool("a", "r1"),
        tool("b", "r2"),
        ai("done"),
    ]
    cap = sc.build_capture(msgs, 1)
    assert cap.n_tool_calls == 2 and cap.tools_used == ["fetch_url", "web_search"]


# ── images ──────────────────────────────────────────────────────────────────


def _png_data_url(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def test_images_become_references_and_bytes_are_kept_once():
    img = _png_data_url(b"\x89PNG-bytes")
    msgs = [
        human([{"type": "text", "text": "<user_query>what is this</user_query>"},
               {"type": "image_url", "image_url": {"url": img}}]),
        ai("A cat."),
        human([{"type": "text", "text": "<user_query>and now?</user_query>"},
               {"type": "image_url", "image_url": {"url": img}}]),  # same image again
        ai("Still a cat."),
    ]
    cap = sc.build_capture(msgs, 3)
    assert cap.has_image
    assert len(cap.images) == 1  # deduped by hash
    (sha, (mime, data)), = cap.images.items()
    assert mime == "image/png" and data == b"\x89PNG-bytes"
    parts = cap.messages[0]["content"]
    assert parts[0] == {"type": "text", "text": "<user_query>what is this</user_query>"}
    assert parts[1] == {"type": "image_url", "image_url": {"url": f"omni-image://{sha}"}}
    assert "base64" not in json.dumps(cap.messages)  # nothing inlined


def test_text_only_list_collapses_to_a_string_like_production_does():
    msgs = [human([{"type": "text", "text": "plain"}]), ai("ok")]
    assert sc.build_capture(msgs, 1).messages[0]["content"] == "plain"


# ── privacy flags ───────────────────────────────────────────────────────────


def test_flags_detect_memory_and_attachments():
    plain = sc.build_capture([human("<user_query>q</user_query>"), ai("a")], 1)
    assert not (plain.has_memory or plain.has_attachments or plain.has_image)

    rich = sc.build_capture(
        [human("<user_memory>\nlives in Urbana\n</user_memory>\n\n<attached_files>x</attached_files>"), ai("a")], 1
    )
    assert rich.has_memory and rich.has_attachments


# ── teacher gating ──────────────────────────────────────────────────────────


def test_model_name_comes_from_response_metadata():
    # The gate depends on langchain-openai's Responses path putting the model in
    # response_metadata["model_name"] (verified in its source, not against a live call).
    cap = sc.build_capture([human("q"), ai("a", meta={"model_name": "gpt-6-luna"})], 1)
    assert cap.turn_models == ["gpt-6-luna"]
    assert sc.build_capture([human("q"), ai("a", meta={"model": "gpt-6-luna"})], 1).turn_models == ["gpt-6-luna"]


def test_only_teacher_turns_are_captured():
    with pytest.raises(sc.CaptureError) as e:
        sc.build_capture([human("q"), ai("a", meta=OTHER)], 1)
    assert e.value.code == "not_teacher_model"


def test_unattributed_turns_fail_closed():
    with pytest.raises(sc.CaptureError) as e:
        sc.build_capture([human("q"), ai("a", meta={})], 1)
    assert e.value.code == "not_teacher_model"


def test_mixed_models_inside_the_thumbed_turn_are_refused():
    msgs = [human("q"), ai(calls=[("c", "web_search", {})], meta=OTHER), tool("c", "r"), ai("a")]
    with pytest.raises(sc.CaptureError) as e:
        sc.build_capture(msgs, 1)
    assert e.value.code == "not_teacher_model"


def test_an_earlier_other_model_turn_is_recorded_for_the_builder_to_filter():
    msgs = [human("one"), ai("a1", meta=OTHER), human("two"), ai("a2")]
    cap = sc.build_capture(msgs, 3)  # thumbed turn is luna, so it records...
    assert cap.turn_models == ["gpt-6-luna-2026-09-01"]
    assert cap.models_seen == ["gemini-3.6-flash", "gpt-6-luna-2026-09-01"]  # ...and says what else is in there


# ── UI / checkpoint alignment ───────────────────────────────────────────────


def test_alignment_passes_when_the_histories_agree():
    msgs = one_search_turn("first") + one_search_turn("second")
    ui = [
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "x"},
        {"role": "user", "content": "second"}, {"role": "assistant", "content": "y"},
    ]
    sc.build_capture(msgs, 1, ui_messages=ui)
    sc.build_capture(msgs, 3, ui_messages=ui)


def test_alignment_catches_an_off_by_one():
    msgs = one_search_turn("first") + one_search_turn("second")
    ui = [
        {"role": "user", "content": "first"}, {"role": "assistant", "content": "x"},
        {"role": "user", "content": "second"}, {"role": "assistant", "content": "y"},
    ]
    # The UI says turn 3 is "second"; if the client had sent the checkpoint's
    # turn 1 for it, the question would not match.
    ui_shifted = ui[2:] + ui[:2]
    with pytest.raises(sc.CaptureError) as e:
        sc.build_capture(msgs, 1, ui_messages=ui_shifted)
    assert e.value.code == "misaligned"
    # Roles that do not form a pair are caught before any text comparison.
    with pytest.raises(sc.CaptureError) as e:
        sc.build_capture(msgs, 1, ui_messages=[{"role": "assistant"}, {"role": "user"}])
    assert e.value.code == "misaligned"


def test_an_unsynced_thread_cannot_be_compared_and_passes():
    sc.build_capture(one_search_turn(), 1, ui_messages=[])


def test_oversized_examples_are_refused(monkeypatch):
    monkeypatch.setattr(sc, "MAX_APPROX_CHARS", 50)
    with pytest.raises(sc.CaptureError) as e:
        sc.build_capture([human("x" * 100), ai("a")], 1)
    assert e.value.code == "too_large"
