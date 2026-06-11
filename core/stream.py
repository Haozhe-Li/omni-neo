"""Unified chat stream handler for the single Omni agent.

`run_agent_stream` is an async generator yielding Server-Sent-Event strings. It
runs the pre-flight widget predictor and the agent concurrently (widgets flush
the instant they are ready, independent of the agent's first token), normalises
LangGraph's stream into a stable wire protocol, and validates artifact/report
tool output before forwarding it.

Wire protocol (one JSON object per `data:` line):
    widget    {type, widget, data}                     – pre-flight live-data card
    reasoning {type, content}                           – status / thinking note
    tool_call {type, tool, args}                        – agent is calling a tool
    tool      {type, tool, content}                     – raw tool result
    sources   {type, sources:[{title,url,content}]}     – accumulated citations
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
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from core.agent import get_agent, SKILL_FILES
from core.prompt_guard import is_harmful
from core.utils.format import _extract_domain_metadata
from core.tools.artifact_tools import ARTIFACT_SENTINEL
from core.widget_predictor import predict_widgets
from core.RAG.file_parser import get_read_presigned_url
from core.database.db_user_files import get_file_record

# Tool names whose calls are represented by artifact events, not tool_call.
_ARTIFACT_TOOL_NAMES = {"render_chart"}

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


def build_message_content(
    query: str, personalization: str, attached_file_ids: list[dict[str, str]] | None
) -> str | list:
    """Build the user message, inlining images and naming attached documents."""
    base_query = f"User Query: {query}\n\nPersonalization: {personalization}\n\n"
    if not attached_file_ids:
        return base_query

    image_blocks: list[dict] = []
    document_names: list[str] = []
    for file_info in attached_file_ids:
        for file_id, filename in file_info.items():
            record = get_file_record(file_id)
            if not record:
                continue
            if record["category"] == "image":
                url = get_read_presigned_url(file_id)
                if url:
                    image_blocks.append({"type": "image_url", "image_url": {"url": url}})
            elif record["category"] == "document":
                document_names.append(filename)

    if document_names:
        base_query = (
            "The user has uploaded these files. Use them via read_user_document when relevant:\n"
            + "\n".join(document_names)
            + "\n\n"
            + base_query
        )

    if image_blocks:
        return [{"type": "text", "text": base_query}, *image_blocks]
    return base_query


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


def _stream_agent(
    query: str,
    thread_id: str | None,
    mode: str,
    personalization: str,
    attached_file_ids: list[dict[str, str]] | None,
):
    """Drive the agent and yield SSE strings (everything except widgets).

    Synchronous because the app uses a sync Postgres checkpointer; it is run in
    a worker thread by ``run_agent_stream``.
    """
    profile = "pro" if mode == "pro" else "fast"
    agent = get_agent(profile)
    config = {"configurable": {"thread_id": thread_id}}
    content = build_message_content(query, personalization, attached_file_ids)

    # Pro is a deep agent with a StateBackend: hand it the skill files so the
    # SkillsMiddleware can surface their metadata and read them on demand.
    input_state: dict = {"messages": [{"role": "user", "content": content}]}
    if profile == "pro" and SKILL_FILES:
        input_state["files"] = SKILL_FILES

    seen_sources: set[tuple] = set()
    all_sources: list[dict] = []
    artifact_ids: list[str] = []
    announced_drafts: set = set()
    produced_text = False

    for raw in agent.stream(
        input_state,
        config=config,
        stream_mode=["messages", "updates"],
        subgraphs=True,
    ):
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
                text = _text_of(chunk.content)
                if text:
                    produced_text = True
                    yield _sse({"type": "text", "content": text})
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

                    # Regular tool → surface citations
                    meta = _extract_domain_metadata(name, raw_content)
                    new_sources = []
                    for s in meta.get("sources", []):
                        key = (s.get("title"), s.get("url"), s.get("content"))
                        if key not in seen_sources:
                            seen_sources.add(key)
                            all_sources.append(s)
                            new_sources.append(s)
                    if new_sources:
                        yield _sse({"type": "sources", "sources": new_sources})

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
):
    """Top-level SSE generator: widgets + agent, concurrent, fail-soft."""
    if is_harmful(query):
        yield _sse({"type": "error", "content": _REFUSAL})
        return

    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    async def widget_producer():
        try:
            for w in await predict_widgets(query):
                await queue.put(_sse({"type": "widget", **w}))
        except Exception as exc:  # widgets must never break the chat
            print(f"[stream] widget producer error: {exc}")
        finally:
            await queue.put(_DONE)

    loop = asyncio.get_running_loop()

    async def agent_producer():
        def run_sync():
            try:
                for event in _stream_agent(
                    query, thread_id, mode, personalization, attached_file_ids
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:
                import traceback

                traceback.print_exc()
                loop.call_soon_threadsafe(
                    queue.put_nowait, _sse({"type": "error", "content": str(exc)})
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        await asyncio.to_thread(run_sync)

    tasks = [asyncio.create_task(widget_producer()), asyncio.create_task(agent_producer())]
    remaining = len(tasks)
    try:
        while remaining:
            item = await queue.get()
            if item is _DONE:
                remaining -= 1
                continue
            yield item
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
