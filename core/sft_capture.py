"""Turn one thumbed-up exchange into a fine-tuning example.

A thumbs-up on a teacher-model answer (luna) records the conversation *as the
agent saw it* — straight out of the LangGraph checkpoint, so every tool call,
every tool result and every image is there — as OpenAI-style messages, which is
the shape `finetune/*/train.py` consumes (`{"messages": [...], "tools": [...]}`).
The system prompt and tool schemas are not stored per example; they are the
harness snapshot the example points at (core/harness_snapshot.py).

This module is pure: LangChain messages in, plain dicts out. No Redis, no
Postgres, no agent — the router fetches the checkpoint and hands the messages
over, which is also what makes this testable offline (tests/test_sft_capture.py).

## What is captured

Everything from the start of the thread through the end of the thumbed turn,
not just the thumbed turn. The turn is the unit of *approval*, but the model
reads the whole history when it answers, and the training loss covers every
assistant message in the row — so earlier turns come along as context and are
trained on too. `models_seen` records which model produced each one so the
dataset builder can refuse rows whose earlier turns came from a different model.

## Turn numbers

The frontend's `turn` for an exchange is the 1-indexed length of its message
list when the user message was added, so it is odd: the K-th user message is
turn 2K-1 (the same rule `_find_rewind_target` in core/routers/chat.py uses).
The thumb lands on the assistant message right after it, whose list index equals
that same number.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

IMAGE_REF_PREFIX = "omni-image://"

# A turn is eligible when every assistant message in it reports a model whose
# name contains one of these. Substring, not equality: providers append dated
# suffixes to the id they were asked for.
TEACHER_MODEL_MARKERS = tuple(
    m.strip().lower()
    for m in os.getenv("SFT_TEACHER_MODEL_MARKERS", "luna").split(",")
    if m.strip()
)

# A row this large is a runaway tool loop or a pasted book, not a useful example,
# and the builder would have to truncate it to nothing anyway.
MAX_APPROX_CHARS = int(os.getenv("SFT_MAX_APPROX_CHARS", "2000000"))

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w.+/-]+);base64,(?P<b64>.+)$", re.S)


class CaptureError(Exception):
    """The exchange cannot be turned into a training example.

    `code` is a short stable string the router returns to the client; the
    message is for logs.
    """

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


@dataclass
class CapturedTurn:
    messages: list[dict]                       # OpenAI-style, no system message
    images: dict[str, tuple[str, bytes]] = field(default_factory=dict)  # sha256 -> (mime, bytes)
    models_seen: list[str] = field(default_factory=list)   # whole captured prefix
    turn_models: list[str] = field(default_factory=list)   # the thumbed turn only
    n_assistant_turns: int = 0
    n_tool_calls: int = 0
    tools_used: list[str] = field(default_factory=list)
    approx_chars: int = 0
    has_image: bool = False
    has_memory: bool = False
    has_attachments: bool = False
    compacted: bool = False


# A small copy of core/stream.py's helper rather than an import of it: that module
# pulls in the agent, every chat model and the data layer, and this one is meant
# to be importable (and testable) with none of them.
def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


# ── locating the turn ───────────────────────────────────────────────────────


def slice_through_turn(messages: list[BaseMessage], turn: int) -> list[BaseMessage]:
    """Messages from the start of the thread through the end of `turn`'s answer.

    `turn` is the frontend's odd, 1-indexed number (see the module docstring).
    Raises CaptureError if the turn does not exist or has not finished — the
    last message must be an assistant answer, not a tool call in flight.
    """
    if turn < 1 or turn % 2 == 0:
        raise CaptureError("bad_turn", f"turn {turn} is not a user-message turn number")
    k = (turn + 1) // 2

    human_idx = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_idx) < k:
        raise CaptureError("turn_not_found", f"thread has {len(human_idx)} user turns, asked for {k}")
    end = human_idx[k] if len(human_idx) > k else len(messages)
    prefix = messages[:end]

    last = prefix[-1]
    if not isinstance(last, AIMessage) or last.tool_calls or not _text_of(last.content).strip():
        raise CaptureError("turn_incomplete", "the turn does not end on a finished assistant answer")
    return prefix


def check_ui_alignment(ui_messages: list[dict], turn: int, prefix: list[BaseMessage]) -> None:
    """Refuse a capture whose turn number points at a different exchange in the
    UI history than in the checkpoint.

    The two histories are kept by different writers (the browser's message list
    and LangGraph's state), so an off-by-one would silently pair an answer with
    the wrong question and then label it as approved. Checked best-effort: a
    thread the UI has not synced yet cannot be compared, and passes.
    """
    if turn >= len(ui_messages):
        return
    user_ui, answer_ui = ui_messages[turn - 1], ui_messages[turn]
    if user_ui.get("role") != "user" or answer_ui.get("role") != "assistant":
        raise CaptureError("misaligned", "UI history has no user/assistant pair at this turn")
    human = next((m for m in reversed(prefix) if isinstance(m, HumanMessage)), None)
    query = " ".join(str(user_ui.get("content") or "").split())
    human_text = " ".join(_human_text(human.content).split()) if human else ""
    if query and query not in human_text:
        raise CaptureError("misaligned", "UI user message is not the checkpoint's user message")


# ── conversion ──────────────────────────────────────────────────────────────


def _human_text(content: Any) -> str:
    return _text_of(content)


def _extract_image(url: str, images: dict[str, tuple[str, bytes]]) -> str:
    """Replace an inline data URL with a stable reference and stash the bytes.

    A remote URL (nothing in the product produces one today) is left alone —
    the example cannot guarantee it still resolves at training time, but there
    are no bytes here to keep either.
    """
    m = _DATA_URL_RE.match(url)
    if not m:
        return url
    try:
        data = base64.b64decode(m.group("b64"), validate=False)
    except (binascii.Error, ValueError):
        return url
    sha = hashlib.sha256(data).hexdigest()
    images[sha] = (m.group("mime"), data)
    return f"{IMAGE_REF_PREFIX}{sha}"


def _user_content(content: Any, images: dict[str, tuple[str, bytes]]) -> tuple[str | list[dict], bool]:
    """-> (OpenAI content, has_image). A text-only list collapses to a string,
    matching how production sends it (`build_message_content`)."""
    if isinstance(content, str):
        return content, False
    parts: list[dict] = []
    has_image = False
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append({"type": "text", "text": block.get("text", "")})
        elif block.get("type") == "image_url":
            raw = block.get("image_url")
            url = raw.get("url") if isinstance(raw, dict) else raw
            if url:
                has_image = True
                parts.append({"type": "image_url", "image_url": {"url": _extract_image(url, images)}})
    if not has_image:
        return "".join(p["text"] for p in parts if p["type"] == "text"), False
    return parts, True


def _model_name(m: AIMessage) -> str | None:
    meta = getattr(m, "response_metadata", None) or {}
    name = meta.get("model_name") or meta.get("model")
    return str(name) if name else None


def convert_messages(prefix: list[BaseMessage], turn_start: int | None = None) -> CapturedTurn:
    """LangChain messages -> `CapturedTurn`. `turn_start` is the index in
    `prefix` of the thumbed turn's user message (default: the last one)."""
    if turn_start is None:
        turn_start = max(i for i, m in enumerate(prefix) if isinstance(m, HumanMessage))

    out: list[dict] = []
    cap = CapturedTurn(messages=out)
    models: set[str] = set()
    turn_models: set[str] = set()
    tools_used: set[str] = set()
    user_texts: list[str] = []

    for i, m in enumerate(prefix):
        if isinstance(m, HumanMessage):
            content, has_image = _user_content(m.content, cap.images)
            cap.has_image = cap.has_image or has_image
            user_texts.append(_text_of(m.content))
            out.append({"role": "user", "content": content})
        elif isinstance(m, AIMessage):
            text = _text_of(m.content).strip()
            if not text and not m.tool_calls:
                continue  # a reasoning-only message: nothing to supervise
            msg: dict[str, Any] = {"role": "assistant", "content": text}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": c.get("id"),
                        "type": "function",
                        "function": {
                            "name": c.get("name"),
                            "arguments": json.dumps(c.get("args") or {}, ensure_ascii=False),
                        },
                    }
                    for c in m.tool_calls
                ]
                cap.n_tool_calls += len(m.tool_calls)
                tools_used.update(c.get("name") or "?" for c in m.tool_calls)
            cap.n_assistant_turns += 1
            if name := _model_name(m):
                models.add(name)
                if i >= turn_start:
                    turn_models.add(name)
            out.append(msg)
        elif isinstance(m, ToolMessage):
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": _text_of(m.content),
            })

    called = {c["id"] for m in out if m["role"] == "assistant" for c in m.get("tool_calls") or []}
    answered = {m["tool_call_id"] for m in out if m["role"] == "tool"}
    if called != answered:
        raise CaptureError(
            "unpaired_tool_calls",
            f"{len(called)} tool calls vs {len(answered)} results — the turn was interrupted mid-tool",
        )

    joined = "\n".join(user_texts)
    cap.has_memory = "<user_memory>" in joined
    cap.has_attachments = "<attached_files>" in joined
    cap.models_seen = sorted(models)
    cap.turn_models = sorted(turn_models)
    cap.tools_used = sorted(tools_used)
    cap.approx_chars = len(json.dumps(out, ensure_ascii=False))
    return cap


def is_teacher_turn(turn_models: list[str]) -> bool:
    """True when the thumbed turn was produced entirely by a teacher model.

    Empty means the checkpoint carried no model attribution at all — treated as
    "not known to be the teacher", so it is not recorded. Failing closed here
    keeps another model's answers out of a dataset that is about luna's.
    """
    return bool(turn_models) and all(
        any(marker in name.lower() for marker in TEACHER_MODEL_MARKERS) for name in turn_models
    )


def build_capture(
    messages: list[BaseMessage],
    turn: int,
    *,
    ui_messages: list[dict] | None = None,
    compacted: bool = False,
) -> CapturedTurn:
    """The whole pipeline for one thumbs-up. Raises CaptureError when the
    exchange should not become an example (the code says why)."""
    prefix = slice_through_turn(messages, turn)
    if ui_messages is not None:
        check_ui_alignment(ui_messages, turn, prefix)
    cap = convert_messages(prefix)
    cap.compacted = compacted
    if cap.approx_chars > MAX_APPROX_CHARS:
        raise CaptureError("too_large", f"{cap.approx_chars:,} chars")
    # Not recorded at all, rather than recorded and filtered at build time:
    # a <user_memory> block holds personal facts and an attachment is the user's
    # own file, and both are meant to get their own dedicated training later.
    # The memory block is injected on a thread's first turn and stays in the
    # history, so every thumbed turn of such a thread is skipped, not just the first.
    if cap.has_memory:
        raise CaptureError("has_memory", "thread carries a <user_memory> block")
    if cap.has_attachments:
        raise CaptureError("has_attachments", "thread carries uploaded files")
    if not is_teacher_turn(cap.turn_models):
        raise CaptureError("not_teacher_model", f"turn served by {cap.turn_models or 'unknown'}")
    return cap
