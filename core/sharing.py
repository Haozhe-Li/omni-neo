"""Turn a live thread into a shareable snapshot, and a snapshot back into state.

Pure: LangChain messages and plain dicts in, plain dicts out. The router reads the
checkpoint and the database and hands the pieces over, which keeps every
privacy-relevant transformation here, in one place, testable offline
(tests/test_sharing.py).

## What a share is

A frozen copy taken at share time. It holds four things: the `ui_messages` the
page renders, the agent state (`messages` + mounted `files`) serialized so a fork
can resume the conversation, the citations, and metadata for the attached files.
It lives only in Postgres — see the `shared_threads` comment in schema.sql for
why that beats copying a Redis thread per share.

## What is removed

Always, no option: the `<user_memory>` block and the `User Location:` line. Both
sit in the first user message of the checkpoint (memory is injected on a thread's
first turn and stays in history; the location rides in every turn's
`<system_reminder>`), neither is visible in the UI, and sharing without removing
them would publish them. Skill files are dropped from the saved `files` too —
every turn re-supplies them (core/stream.py), and saving them would pin a stale
copy of a skill into the fork.

## What is NOT removed

Attachments. A shared thread carries the documents and images in its state, so
whoever forks it can have the model quote them. That is a product decision, not
an oversight (the share dialog says so); it is why this module does not try to
scrub file content. Likewise the sharer's own words, tool results and answers.
What the model says about the sharer (an answer that mentions their city, say)
is not scrubbed either — there is no reliable way to.
"""
from __future__ import annotations

import copy
import json
import os
import re
import secrets
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

MAX_SNAPSHOT_BYTES = int(os.getenv("SHARE_MAX_BYTES", str(10 * 1024 * 1024)))
MAX_ACTIVE_SHARES = int(os.getenv("SHARE_MAX_ACTIVE", "20"))
SKILLS_PREFIX = "/skills/"

SHARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")

_MEMORY_BLOCK_RE = re.compile(r"<user_memory>.*?</user_memory>[ \t]*\n*", re.S)
_LOCATION_LINE_RE = re.compile(r"^User Location:.*\n?", re.M)


class ShareError(Exception):
    """The thread cannot be shared. `code` is a short stable string the router
    maps to an HTTP status; the message is for logs and the client."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


@dataclass
class Snapshot:
    ui_messages: list[dict]
    agent_state: dict
    citations: list[dict]
    n_messages: int
    size_bytes: int
    files_meta: list[dict] = field(default_factory=list)


def new_share_id() -> str:
    """16 URL-safe characters (~96 bits) — long enough that a link cannot be guessed."""
    return secrets.token_urlsafe(12)


# ── privacy scrub ───────────────────────────────────────────────────────────


def scrub_text(text: str) -> str:
    """Remove the memory block and the location line from one message's text."""
    return _LOCATION_LINE_RE.sub("", _MEMORY_BLOCK_RE.sub("", text))


def _scrub_human_dict(d: dict) -> dict:
    data = d.get("data", {})
    content = data.get("content")
    if isinstance(content, str):
        data["content"] = scrub_text(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                block["text"] = scrub_text(block["text"])
    return d


# ── state (de)serialization ─────────────────────────────────────────────────


def _jsonable(obj: Any) -> Any:
    """Round-trip through JSON so what is stored is exactly what is read back."""
    return json.loads(json.dumps(obj, ensure_ascii=False, default=str))


def serialize_state(messages: list[BaseMessage], files: dict | None) -> dict:
    """Agent state -> JSON-safe dict, scrubbed. Human messages only are scrubbed;
    everything else is stored as the checkpoint holds it."""
    as_dicts = [
        _scrub_human_dict(d) if d.get("type") == "human" else d
        for d in _jsonable(messages_to_dict(messages))
    ]
    kept_files = {
        path: data for path, data in (files or {}).items() if not path.startswith(SKILLS_PREFIX)
    }
    return {"messages": as_dicts, "files": _jsonable(kept_files)}


def deserialize_state(blob: dict) -> tuple[list[BaseMessage], dict]:
    """The inverse of `serialize_state`: `(messages, files)` ready for `aupdate_state`."""
    return messages_from_dict(blob.get("messages", [])), blob.get("files", {})


# ── attachments ─────────────────────────────────────────────────────────────


def attachment_ids(ui_messages: list[dict]) -> list[str]:
    """File ids of every attachment chip in the UI history, in order, unique."""
    seen: dict[str, None] = {}
    for m in ui_messages:
        for f in m.get("attachedFiles") or []:
            if isinstance(f, dict) and f.get("id"):
                seen.setdefault(f["id"], None)
    return list(seen)


def remap_attachment_ids(ui_messages: list[dict], mapping: dict[str, str]) -> list[dict]:
    """A copy of `ui_messages` with each attachment's file id replaced by its new
    one (ids the mapping does not know are left alone)."""
    out = copy.deepcopy(ui_messages)
    for m in out:
        for f in m.get("attachedFiles") or []:
            if isinstance(f, dict) and f.get("id") in mapping:
                f["id"] = mapping[f["id"]]
    return out


# ── the snapshot ────────────────────────────────────────────────────────────


def build_snapshot(
    *,
    ui_messages: list[dict],
    messages: list[BaseMessage],
    files: dict | None,
    citations: list[dict],
    files_meta: list[dict] | None = None,
) -> Snapshot:
    """Assemble a share from a thread's pieces. Raises ShareError if there is
    nothing to share or it is too big."""
    if not ui_messages or not messages:
        raise ShareError("empty_thread", "nothing to share yet")
    state = serialize_state(messages, files)
    ui = _jsonable(ui_messages)
    cites = _jsonable(citations)
    meta = _jsonable(files_meta or [])
    size = sum(len(json.dumps(x, ensure_ascii=False)) for x in (ui, state, cites, meta))
    if size > MAX_SNAPSHOT_BYTES:
        raise ShareError(
            "too_large",
            f"{size / 1e6:.1f} MB exceeds the {MAX_SNAPSHOT_BYTES / 1e6:.0f} MB limit",
        )
    return Snapshot(
        ui_messages=ui,
        agent_state=state,
        citations=cites,
        n_messages=len(ui),
        size_bytes=size,
        files_meta=meta,
    )
