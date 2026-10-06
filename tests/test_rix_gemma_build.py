"""finetune/rix_gemma/build_dataset.py end to end, with the database stubbed out.

    venv/bin/python3.12 -m pytest tests/test_rix_gemma_build.py -q

`fetch_examples` / `fetch_images` / `current_harness_hash` are replaced with
in-memory rows shaped like the SQL result, so this exercises the real filtering,
prefix de-duplication, lead-in splitting, truncation and JSONL writing — but not
the queries themselves (those need a Postgres).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("rix_build", ROOT / "finetune" / "rix_gemma" / "build_dataset.py")
build = importlib.util.module_from_spec(spec)
sys.modules["rix_build"] = build
spec.loader.exec_module(build)

HASH = "abc123"
TOOLS = [{"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}]


def example(id_, thread="t1", turn=1, user="u1", messages=None, **over):
    base = {
        "id": id_, "thread_id": thread, "turn": turn, "user_id": user, "harness_hash": HASH,
        "models_seen": ["gpt-6-luna"], "has_image": False, "has_memory": False,
        "has_attachments": False, "compacted": False, "tools_used": ["web_search"],
        "system_prompt": "SYSTEM", "tools": TOOLS,
        "messages": messages or [
            {"role": "user", "content": f"q{id_}"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": f"c{id_}", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{id_}", "content": "result"},
            {"role": "assistant", "content": f"answer {id_}"},
        ],
    }
    base.update(over)
    return base


@pytest.fixture
def run(monkeypatch, tmp_path):
    def _run(rows, *argv):
        monkeypatch.setattr(build, "fetch_examples", lambda statuses: rows)
        monkeypatch.setattr(build, "current_harness_hash", lambda: HASH)
        monkeypatch.setattr(build, "load_tokenizer", lambda: (lambda s: len(s) // 3))
        monkeypatch.setattr(build, "DATA", tmp_path)
        monkeypatch.setattr(sys, "argv", ["build_dataset.py", *argv])
        code = build.main()
        out = tmp_path / "sft_train.jsonl"
        lines = out.read_text().split("\n") if out.exists() else []
        return code, [json.loads(line) for line in lines if line.strip()], tmp_path
    return _run


def test_row_shape_matches_what_train_py_consumes(run):
    code, rows, _ = run([example(1)])
    assert code == 0 and len(rows) == 1
    msgs = rows[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "SYSTEM"}
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant"]
    assert rows[0]["tools"] == TOOLS
    assert set(rows[0]) == {"messages", "tools"}


def test_filters_and_their_reasons(run, capsys):
    rows = [
        example(1),
        example(2, thread="t2", models_seen=["gpt-6-luna", "gemini-3.6-flash"]),
        example(3, thread="t3", models_seen=[]),                      # unattributed fails closed
        example(4, thread="t4", has_memory=True),
        example(5, thread="t5", has_attachments=True),
        example(6, thread="t6", has_image=True),
        example(7, thread="t7", compacted=True),
        example(8, thread="t8", harness_hash="old"),
    ]
    code, out, tmp = run(rows)
    assert code == 0
    assert [r["messages"][1]["content"] for r in out] == ["q1"]
    report = capsys.readouterr().out
    for reason in ("non-teacher", "has <user_memory>", "has attachments", "has image", "summarised", "other harness"):
        assert reason in report
    assert json.loads((tmp / "manifest.json").read_text())["train_example_ids"] == [1]


def test_overrides_let_filtered_rows_through(run):
    rows = [example(1, has_memory=True), example(2, thread="t2", has_attachments=True)]
    _, out, _ = run(rows, "--include-memory", "--include-attachments")
    assert len(out) == 2


def test_teacher_filter_can_be_disabled(run):
    _, out, _ = run([example(1, models_seen=["gemini-3.6-flash"])], "--teacher", "")
    assert len(out) == 1


def test_stale_harness_only_input_fails_loudly(run):
    code, out, _ = run([example(1, harness_hash="old")])
    assert code == 1 and out == []


def test_a_thread_thumbed_twice_yields_only_the_longer_row(run, capsys):
    t1 = example(1, turn=1)["messages"]
    t3 = t1 + [{"role": "user", "content": "follow up"}, {"role": "assistant", "content": "second answer"}]
    rows = [example(1, turn=1, messages=t1), example(2, turn=3, messages=t3)]
    _, out, tmp = run(rows)
    assert len(out) == 1 and out[0]["messages"][-1]["content"] == "second answer"
    assert json.loads((tmp / "manifest.json").read_text())["train_example_ids"] == [2]
    assert "superseded" in capsys.readouterr().out


def test_different_threads_with_the_same_text_are_not_collapsed(run):
    _, out, _ = run([example(1, thread="a"), example(1, thread="b")])
    assert len(out) == 2


def test_max_per_user(run):
    rows = [example(i, thread=f"t{i}", user="same") for i in range(1, 5)] + [example(9, thread="o", user="other")]
    _, out, _ = run(rows, "--max-per-user", "2")
    assert len(out) == 3


def test_lead_in_text_is_split_off_its_tool_call_for_gemma():
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "Let me check.", "tool_calls": [{"id": "c", "type": "function",
         "function": {"name": "web_search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "r"},
        {"role": "assistant", "content": "done"},
    ]
    out = build.split_lead_ins(msgs)
    assert [(m["role"], m["content"], bool(m.get("tool_calls"))) for m in out] == [
        ("user", "q", False),
        ("assistant", "Let me check.", False),
        ("assistant", "", True),
        ("tool", "r", False),
        ("assistant", "done", False),
    ]


def test_oversized_tool_results_are_truncated_to_the_cap_with_a_marker(run):
    big = example(1, messages=[
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "x" * 60_000},
        {"role": "assistant", "content": "answer"},
    ])
    _, out, _ = run([big], "--cap", "2000")
    tool_msg = next(m for m in out[0]["messages"] if m["role"] == "tool")
    assert len(tool_msg["content"]) < 60_000 and tool_msg["content"].endswith("…[内容已截断]")
    assert out[0]["messages"][-1]["content"] == "answer"  # the answer itself is never cut


def test_images_are_dropped_by_default_and_rehydrated_when_kept(run, monkeypatch):
    img_msgs = [
        {"role": "user", "content": [{"type": "text", "text": "q"},
                                     {"type": "image_url", "image_url": {"url": "omni-image://deadbeef"}}]},
        {"role": "assistant", "content": "a cat"},
    ]
    row = example(1, messages=img_msgs, has_image=True, tools_used=[])
    assert run([row])[1] == []                                  # default: dropped

    monkeypatch.setattr(build, "fetch_images", lambda shas: {"deadbeef": "data:image/png;base64,AAAA"})
    _, out, _ = run([row], "--images", "keep")
    assert out[0]["messages"][1]["content"][1]["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_missing_image_bytes_skip_the_row_instead_of_training_on_a_dangling_ref(run, monkeypatch):
    img_msgs = [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "omni-image://gone"}}]},
        {"role": "assistant", "content": "a"},
    ]
    monkeypatch.setattr(build, "fetch_images", lambda shas: {})
    _, out, _ = run([example(1, messages=img_msgs, has_image=True)], "--images", "keep")
    assert out == []


def test_holdout_splits_off_rows(run):
    rows = [example(i, thread=f"t{i}") for i in range(1, 6)]
    code, out, tmp = run(rows, "--holdout", "2")
    assert code == 0 and len(out) == 3
    assert len((tmp / "sft_holdout.jsonl").read_text().strip().split("\n")) == 2


def test_report_mode_writes_nothing(run):
    code, out, tmp = run([example(1)], "--report")
    assert code == 0 and out == []
    assert not list(tmp.iterdir())
