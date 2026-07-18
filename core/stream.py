"""Unified chat stream handler for the single Omni agent.

`run_agent_stream` is an async generator yielding Server-Sent-Event strings. It
runs the pre-flight widget predictor and the agent concurrently (widgets flush
the instant they are ready, independent of the agent's first token), normalises
LangGraph's stream into a stable wire protocol, and validates artifact/report
tool output before forwarding it.

Wire protocol (one JSON object per `data:` line):
    widget    {type, widget, data}                     – pre-flight live-data card
    reasoning {type, content}                           – streamed reasoning/thinking
                                                            tokens (fast: Groq gpt-oss
                                                            reasoning_content; pro:
                                                            Gemini thinking blocks)
    tool_call {type, tool, args}                        – agent is calling a tool
    tool      {type, tool, content}                     – raw tool result
    sources   {type, sources:[{n,title,url,content}]}    – accumulated citations;
                                                            `n` matches the [n]
                                                            markers in the text
    text      {type, content}                           – streamed answer token(s)
    artifact  {type, id, title, kind, spec}             – chart for the side panel
    done      {type, sources, artifacts}                – terminal summary
    error     {type, content}

Reports are NOT a distinct event: the agent writes them inline as a
`<report>…</report>` block within the normal `text` stream (just like charts are
written inline as ```echarts fences). The frontend parses the block out of the
answer and renders it live in the side reader.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langsmith import tracing_context
from deepagents.backends.utils import create_file_data

from core.agent import get_agent, FAST_SKILL_FILES, PRO_SKILL_FILES
from core.prompt_guard import is_harmful
from core.utils.citations import all_citations, reset_citation_registry_async, register_document_citation
from core.tools.artifact_tools import ARTIFACT_SENTINEL
from core.widget_predictor import predict_widgets
from core.RAG.file_parser import get_image_base64_data_url, MARKDOWN_SOURCE_TYPES
from core.database.db_user_files import get_file_record, count_prior_ready_files_with_name

# Tool names whose calls are represented by artifact events, not tool_call.
_ARTIFACT_TOOL_NAMES = {"render_chart"}

# A single citation marker in any bracket style the model might emit.
_CITE_MARKER = r"(?:\[\d+\]|［\d+］|【\d+】)"

# Kill switch for citation *reordering* (the cluster merge below and the
# aggressive hold-back that serves it). Off by default: the model's marker
# placement streams through verbatim, and only an unclosed opener ("[1") is
# held so a marker is never split across two SSE events. Set CITATION_NORM=true
# to enable the merge. Full-width -> ASCII marker conversion ([1] -> [1]) is
# NOT gated — the frontend only links the ASCII form.
CITATION_NORM = os.getenv("CITATION_NORM", "false").strip().lower() in {
    "1", "true", "yes", "on"
}

# Held back at the end of the buffer so a marker never leaks to the frontend
# half-written or in the wrong position:
#  - an unclosed opener, e.g. "...as reported" + "[1" — the close might be in
#    the next chunk.
#  - a run of *closed* markers, each optionally trailed by sentence punctuation
#    and whitespace, e.g. "...noted[2]" or "...noted[2]. " or "...noted[2].\n"
#    — the next chunk might bring the punctuation (which must land BEFORE the
#    marker) and/or another marker for the same paragraph-end cluster (which
#    must merge into the same stack, per <citation_policy>). Once a chunk is
#    flushed as its own SSE event neither fix-up can happen anymore, so all of
#    this is held until real content (not marker/punct/whitespace) shows up.
#
# The two alternatives must compose: a closed run FOLLOWED BY an unclosed
# opener ("[6][", "[6][8]。[") is one hold, not a hold of just the opener.
# Tokenizers routinely emit "][" as a single token, so the buffer passes
# through "...[6][" constantly; matching only the trailing "[" there flushes
# "[6]" on its own — stranding it before punctuation that hasn't arrived yet
# and producing the "claim[6]。[8][9]" split this whole regex exists to
# prevent.
_TRAILING_CITE_RE = (
    re.compile(
        rf"(?:(?:{_CITE_MARKER})+\s*[.!?。！？]?\s*)*"
        rf"(?:[\[［【]\d*|(?:{_CITE_MARKER})+\s*[.!?。！？]?\s*)$"
    )
    if CITATION_NORM
    # Merge disabled: hold only an unclosed opener. Holding punctuation and
    # closed runs would add latency for a fix-up that will never run.
    else re.compile(r"[\[［【]\d*$")
)

# Closed full-width citation markers to normalize to the ASCII form the
# frontend actually matches on: 【1】 / [1] -> [1].
_FULLWIDTH_CITE_RE = re.compile(r"[［【](\d+)[］】]")

# A citation marker (or run of stacked markers, [1][2]) sitting immediately
# before sentence-ending punctuation, plus any further markers trailing after
# it separated only by whitespace/newlines (no real content in between — the
# model split one paragraph-end cluster across a line break), all get merged
# into one stack right after the punctuation:
# "claim[1]." -> "claim.[1]"; "claim[1].\n[2]" -> "claim.[1][2]".
# Only sentence-enders qualify (., !, ?, 。, ！, ？) — commas and other
# mid-sentence punctuation are left alone. The `\s*` between the first marker
# run and the punctuation tolerates a stray space the model sometimes emits
# there (e.g. "claim[1] ." — otherwise "[1]" and "." never merge with a
# trailing "[2]", producing exactly the "claim[1] .[2]" split the frontend
# renders as two citation chips with the period stuck between them).
_CITE_CLUSTER_RE = re.compile(
    rf"((?:\[\d+\])+)\s*([.!?。！？])((?:\s*(?:\[\d+\])+[.!?。！？]?)*)"
)


def _merge_citation_cluster(m: "re.Match[str]") -> str:
    punct = m.group(2)
    markers = re.findall(r"\[\d+\]", m.group(1) + m.group(3))
    return punct + "".join(markers)


def _normalize_citations(text: str) -> str:
    text = _FULLWIDTH_CITE_RE.sub(r"[\1]", text)
    if not CITATION_NORM:
        return text
    return _CITE_CLUSTER_RE.sub(_merge_citation_cluster, text)

_REFUSAL = "I’m sorry, but I can’t help with that."


def _sse(obj: dict[str, Any]) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _text_of(content: Any) -> str:
    """Coerce a message's `content` (str or content-block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _reasoning_of(chunk: AIMessageChunk) -> str:
    """Pull streamed reasoning/thinking text off an AIMessageChunk, if any.

    Groq (fast, reasoning_format="parsed") surfaces it as a plain string in
    additional_kwargs.reasoning_content. Gemini (pro, include_thoughts=True)
    surfaces it as {"type": "thinking", "thinking": ...} blocks inside
    `content` alongside the regular text blocks.
    """
    reasoning = chunk.additional_kwargs.get("reasoning_content")
    if reasoning:
        return reasoning
    content = chunk.content
    if isinstance(content, list):
        return "".join(
            part.get("thinking", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "thinking"
        )
    return ""


def _dedupe_document_name(
    thread_id: str | None, filename: str, file_id: str, created_at, file_type: str
) -> str:
    """Finder-style name collision handling: name.ext, name(1).ext, name(2).ext, ...

    Rank is the count of other ready same-named files in the thread ordered
    strictly before this one (by created_at, file_id) — a single DB-derived
    total order, so files attached in the same batch are ranked correctly
    against each other without double-counting. Ranking is keyed to the
    original filename/extension (as stored in Postgres) even though binary
    formats get an ``.md`` display extension below, since that's what was
    actually re-uploaded.
    """
    display_name = filename
    if file_type in MARKDOWN_SOURCE_TYPES:
        stem, ext = os.path.splitext(filename)
        # Fold the original extension into the stem (report.pdf -> report_pdf.md)
        # so a same-named .pdf and .docx don't collide once both become .md.
        display_name = f"{stem}_{ext.lstrip('.')}.md" if ext else f"{stem}.md"

    if not thread_id:
        return display_name
    rank = count_prior_ready_files_with_name(thread_id, filename, file_id, created_at)
    if rank == 0:
        return display_name
    stem, ext = os.path.splitext(display_name)
    return f"{stem}({rank}){ext}"


def build_message_content(
    query: str,
    personalization: str,
    attached_file_ids: list[dict[str, str]] | None,
    thread_id: str | None = None,
) -> tuple[str | list, dict, list[dict]]:
    """Build the user message, inlining images and mounting documents as files.

    Returns ``(content, files, doc_sources)``. ``files`` holds virtual-path ->
    FileData entries for ready documents, to be merged into the agent's
    filesystem state (``input_state["files"]``) so it can `read_file`/`grep`
    them. ``doc_sources`` carries one citation record per newly-mounted
    document (``{n, title, url: "", content}``, `url` always empty — the
    frontend treats that as "uploaded document, not a link"). A document is
    never read via a tool call, so it never flows through the ToolMessage
    citation path — the caller must fold `doc_sources` into the `sources` SSE
    stream itself.

    Must be called after `reset_citation_registry(thread_id, turn)` —
    `register_document_citation` below needs that context already set up.
    """
    base_query = f"User Query: {query}\n\nPersonalization: {personalization}\n\n"
    if not attached_file_ids:
        return base_query, {}, []

    image_blocks: list[dict] = []
    document_notes: list[str] = []
    files: dict[str, dict] = {}
    doc_sources: list[dict] = []
    for file_info in attached_file_ids:
        for file_id, filename in file_info.items():
            record = get_file_record(file_id)
            if not record:
                continue
            if record["category"] == "image":
                data_url = get_image_base64_data_url(file_id)
                if data_url:
                    image_blocks.append({"type": "image_url", "image_url": {"url": data_url}})
            elif record["category"] == "document":
                if record["status"] == "failed":
                    document_notes.append(f"{filename} (failed to process — ask the user to re-upload it)")
                    continue
                if record["status"] != "ready":
                    document_notes.append(f"{filename} (still being processed, not readable yet)")
                    continue
                mounted_name = _dedupe_document_name(
                    thread_id, filename, file_id, record["created_at"], record["file_type"]
                )
                path = f"/uploads/{mounted_name}"
                full_text = record.get("extracted_text") or ""
                files[path] = create_file_data(full_text)
                n = register_document_citation(
                    title=filename,
                    file_id=file_id,
                    content=full_text[:1000],
                    index_content=full_text[:10000],
                )
                doc_sources.append({"n": n, "title": filename, "url": "", "content": full_text[:1000]})
                document_notes.append(f"{filename} -> mounted at {path}, cite as [{n}] when you use it")

    if document_notes:
        base_query = (
            "The user has uploaded these files, available in your filesystem. "
            "Use `ls`, `read_file`, or `grep` to read them:\n"
            + "\n".join(document_notes)
            + "\n\n"
            + base_query
        )

    if image_blocks:
        return [{"type": "text", "text": base_query}, *image_blocks], files, doc_sources
    return base_query, files, doc_sources


def _normalize_stream_item(item: Any) -> tuple[str | None, Any]:
    """Reduce a LangGraph stream item to a ``(mode, data)`` pair.

    Handles the ``(mode, data)`` shape and the ``subgraphs=True`` variants
    ``(namespace, (mode, data))`` and ``(namespace, mode, data)``.
    """
    if not isinstance(item, tuple):
        return None, item
    if len(item) == 3:  # (namespace, mode, data)
        return item[1], item[2]
    if len(item) == 2:
        first, second = item
        if isinstance(first, str):  # (mode, data)
            return first, second
        if isinstance(second, tuple) and len(second) == 2 and isinstance(second[0], str):
            return second[0], second[1]  # (namespace, (mode, data))
    return None, item


async def _stream_agent(
    query: str,
    thread_id: str | None,
    mode: str,
    personalization: str,
    attached_file_ids: list[dict[str, str]] | None,
    *,
    turn: int | None = None,
    rewind_config: dict | None = None,
    extra_sources: list[dict] | None = None,
    cancellation_event: asyncio.Event | None = None,
):
    """Drive the agent and yield SSE strings (everything except widgets).

    If ``rewind_config`` is provided the agent replays / forks from that
    LangGraph checkpoint instead of appending a new user message. In that
    case ``build_message_content`` isn't called here — the rewind endpoint
    already called it itself (to fold a possible new attachment into the
    forked checkpoint) — so any document citations it produced are passed in
    via ``extra_sources`` instead of being computed fresh below.
    """
    profile = "pro" if mode == "pro" else "fast"
    agent = get_agent(profile)

    # Must run before build_message_content: mounting a document assigns it a
    # citation number via register_document_citation, which needs the
    # thread/turn context this sets up.
    await reset_citation_registry_async(thread_id, turn)

    if rewind_config is not None:
        # Time-travel: replay from (possibly forked) checkpoint — no new input.
        config = rewind_config
        input_state = None
        doc_sources: list[dict] = []
    else:
        config = {"configurable": {"thread_id": thread_id}}
        content, doc_files, doc_sources = await asyncio.to_thread(
            build_message_content, query, personalization, attached_file_ids, thread_id
        )
        # Both profiles are deep agents with a StateBackend: hand them the skill
        # files (so SkillsMiddleware can surface/read them) plus any uploaded
        # documents mounted this turn (so FilesystemMiddleware's read_file/grep
        # can see them). The files channel merges additively across turns, so
        # documents mounted once stay visible for the rest of the thread.
        input_state = {"messages": [{"role": "user", "content": content}]}
        skill_files = PRO_SKILL_FILES if profile == "pro" else FAST_SKILL_FILES
        files = {**skill_files, **doc_files}
        if files:
            input_state["files"] = files

    seen_sources: set[tuple] = set()
    # Citation numbers already surfaced to the frontend — seeded from whatever
    # `reset_citation_registry`/`build_message_content` already registered
    # (e.g. mounted documents) so the tool-result loop below doesn't re-emit
    # them a second time once it starts polling the registry itself.
    seen_citation_ns: set[int] = {
        c["n"] for c in all_citations() if c.get("n") is not None
    }
    all_sources: list[dict] = []
    artifact_ids: list[str] = []
    announced_drafts: set = set()
    produced_text = False
    pending_text = ""  # holds back a trailing unclosed "[n" citation marker

    # Documents never go through a tool call, so they never hit the
    # ToolMessage-sourced citation path below — surface them the same way
    # tool-fetched sources are: dedup, accumulate, and push an SSE event now.
    new_doc_sources = []
    for s in [*doc_sources, *(extra_sources or [])]:
        key = (s.get("title"), s.get("url"), s.get("content"))
        if key not in seen_sources:
            seen_sources.add(key)
            all_sources.append(s)
            new_doc_sources.append(s)
    if new_doc_sources:
        yield _sse({"type": "sources", "sources": new_doc_sources})

    with tracing_context(project_name=profile):
        async for raw in agent.astream(
            input_state,  # None on rewind → replay from checkpoint
            config=config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            if cancellation_event and cancellation_event.is_set():
                return

            mode_name, data = _normalize_stream_item(raw)

            # ── streamed answer tokens ──────────────────────────────────────────
            if mode_name == "messages":
                chunk = data[0] if isinstance(data, tuple) else data
                if isinstance(chunk, AIMessageChunk):
                    # Detect an artifact/report tool call the moment the model starts
                    # emitting it, so the UI can show "drafting…" before it finishes.
                    for tcc in getattr(chunk, "tool_call_chunks", None) or []:
                        name = tcc.get("name")
                        if name in _ARTIFACT_TOOL_NAMES:
                            key = (name, tcc.get("index"))
                            if key not in announced_drafts:
                                announced_drafts.add(key)
                                yield _sse({"type": "drafting", "tool": name})
                    reasoning = _reasoning_of(chunk)
                    if reasoning:
                        yield _sse({"type": "reasoning", "content": reasoning})
                    text = _text_of(chunk.content)
                    if text:
                        produced_text = True
                        pending_text += text
                        m = _TRAILING_CITE_RE.search(pending_text)
                        safe, pending_text = (
                            (pending_text[: m.start()], pending_text[m.start() :])
                            if m
                            else (pending_text, "")
                        )
                        if safe:
                            yield _sse({"type": "text", "content": _normalize_citations(safe)})
                continue

            # ── tool calls, tool results, artifacts, reports, sources ───────────
            if mode_name != "updates" or not isinstance(data, dict):
                continue

            for node_output in data.values():
                if not isinstance(node_output, dict):
                    continue
                messages = node_output.get("messages")
                if messages is None:
                    continue
                if not isinstance(messages, list):
                    messages = [messages]

                for msg in messages:
                    # Agent's intent to call a (non-artifact) tool
                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            name = tc.get("name")
                            if not name or name in _ARTIFACT_TOOL_NAMES:
                                continue
                            yield _sse(
                                {"type": "tool_call", "tool": name, "args": tc.get("args", {})}
                            )

                    # Tool results: artifact / report sentinels, else sources
                    if isinstance(msg, ToolMessage):
                        name = getattr(msg, "name", "") or ""
                        raw_content = msg.content

                        if name in _ARTIFACT_TOOL_NAMES:
                            parsed = None
                            if isinstance(raw_content, str):
                                try:
                                    parsed = json.loads(raw_content)
                                except json.JSONDecodeError:
                                    parsed = None
                            if isinstance(parsed, dict) and ARTIFACT_SENTINEL in parsed:
                                art = parsed[ARTIFACT_SENTINEL]
                                artifact_ids.append(art["id"])
                                yield _sse({"type": "artifact", **art})
                            # else: the tool returned a validation error → leave it
                            # in the model's context so it can retry; nothing to emit.
                            continue

                        # Regular tool → surface any citations it just registered.
                        # Read from the citation registry (`register_citation`,
                        # called from inside the retrieval tools themselves),
                        # not the tool's own return value — the two can now
                        # differ: junk sources are registered for the record
                        # but deliberately withheld from what the agent reads
                        # back, so parsing the tool's return value would have
                        # silently dropped them from the frontend too.
                        new_sources = []
                        for record in all_citations():
                            n = record.get("n")
                            if n is None or n in seen_citation_ns:
                                continue
                            seen_citation_ns.add(n)
                            src = {
                                "title": record.get("title", ""),
                                "url": record.get("url", ""),
                                "content": record.get("content", ""),
                                "n": n,
                            }
                            if record.get("credibility") is not None:
                                src["credibility"] = record["credibility"]
                            new_sources.append(src)
                        if new_sources:
                            all_sources.extend(new_sources)
                            yield _sse({"type": "sources", "sources": new_sources})

        if pending_text:
            yield _sse({"type": "text", "content": _normalize_citations(pending_text)})

        if not produced_text and not artifact_ids:
            yield _sse({"type": "error", "content": "The agent produced no output."})

        yield _sse(
            {
                "type": "done",
                "sources": all_sources,
                "artifacts": artifact_ids,
            }
        )


async def run_agent_stream(
    query: str,
    thread_id: str | None,
    mode: str = "fast",
    personalization: str = "",
    attached_file_ids: list[dict[str, str]] | None = None,
    user_location: str | None = None,
    user_local_datetime: str | None = None,
    turn: int | None = None,
    rewind_config: dict | None = None,
    extra_sources: list[dict] | None = None,
    cancellation_event: asyncio.Event | None = None,
):
    """Top-level SSE generator: widgets + agent, concurrent, fail-soft."""
    if await is_harmful(query):
        yield _sse({"type": "error", "content": _REFUSAL})
        return

    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _DONE = object()

    async def widget_producer():
        if rewind_config is not None:
            # Rewind replays from a checkpoint — no new query to predict widgets for.
            await queue.put(_DONE)
            return
        try:
            for w in await predict_widgets(
                query,
                user_location=user_location,
                user_local_datetime=user_local_datetime,
            ):
                await queue.put(_sse({"type": "widget", **w}))
        except Exception as exc:  # widgets must never break the chat
            print(f"[stream] widget producer error: {exc}")
        finally:
            await queue.put(_DONE)

    async def agent_producer():
        try:
            async for event in _stream_agent(
                query, thread_id, mode, personalization, attached_file_ids,
                turn=turn,
                rewind_config=rewind_config,
                extra_sources=extra_sources,
                cancellation_event=cancellation_event,
            ):
                await queue.put(event)
        except Exception as exc:
            import traceback

            traceback.print_exc()
            await queue.put(_sse({"type": "error", "content": str(exc)}))
        finally:
            await queue.put(_DONE)

    tasks = [asyncio.create_task(widget_producer()), asyncio.create_task(agent_producer())]
    remaining = len(tasks)
    # Hold the agent's terminal `done` event until BOTH producers have finished.
    # The frontend stops reading the stream the instant it sees `done`, so emitting
    # it while the (slower) widget predictor is still running would drop late
    # widgets — e.g. the entity card, which makes two Serper calls.
    final_done: str | None = None
    try:
        while remaining:
            item = await queue.get()
            if item is _DONE:
                remaining -= 1
                continue
            if cancellation_event and cancellation_event.is_set():
                yield _sse({"type": "stopped"})
                return
            if '"type": "done"' in item:
                final_done = item
                continue
            yield item
        if final_done is not None:
            yield final_done
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
