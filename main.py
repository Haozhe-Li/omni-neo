import dotenv

dotenv.load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import ast
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from core.light_agent import omni_light_agent, LIGHT_AGENT_SYSTEM_PROMPT

# Import the agent and formatter from existing codebase
from core.supervisor import agent, supervisor_system_prompt
from core.utils.format import format_answer
from core.get_title import get_title, get_title_llm_system_prompt
from core.auto_select_model import (
    get_auto_select_model,
    llm_system_prompt as auto_select_model_system_prompt,
)
from core.source_checker import check_source
from core.prompt_guard import is_harmful, has_prompt_leakage, register_sensitive_prompts
from core.utils.data_model import (
    QueryRequest,
    CheckSourceRequest,
    Personalization,
    UpdateMemoriesRequest,
)
from core.auth import (
    get_current_user,
    get_current_user_with_rate_limit,
    get_current_user_check_rate_limit,
    get_optional_user,
    GUEST_DAILY_LIMIT,
)
from core.database.db_user_threads import (
    get_guest_usage_today as _get_guest_usage_today,
)
from core.database.db_user_threads import (
    get_threads_for_user,
    get_thread_messages,
    upsert_thread_messages,
    register_thread,
    update_thread_title,
    delete_user_thread,
    pin_user_thread,
    merge_guest_to_user,
    count_user_threads,
    GUEST_MAX_THREADS,
)
from core.database.db_threads_control import (
    upsert_thread,
    touch_thread,
    cleanup_old_threads,
    delete_thread as delete_thread_state,
    reassign_threads_user,
    pin_thread as pin_thread_state,
    get_thread_owner,
)
from core.utils.utils import format_personalization
from core.memories_update_llm import (
    get_update_memories,
    llm_system_prompt as memories_update_system_prompt,
)
from core.audio_sst import get_text_from_audio
from core.research_helper import omni_research_helper, RESEARCH_HELPER_SYSTEM_PROMPT

app = FastAPI(title="Omni Agent API")

register_sensitive_prompts(
    [
        supervisor_system_prompt,
        LIGHT_AGENT_SYSTEM_PROMPT,
        RESEARCH_HELPER_SYSTEM_PROMPT,
        auto_select_model_system_prompt,
        get_title_llm_system_prompt,
        memories_update_system_prompt,
    ]
)

# Thread pool for fire-and-forget blocking DB calls
_db_executor = ThreadPoolExecutor(max_workers=4)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _assert_thread_access(thread_id: str | None, user_id: str) -> None:
    """
    Verify the requesting user is allowed to access the given thread.
    Raises HTTP 403 if the thread is claimed by a *different* user.
    Unclaimed threads (owner is None) are accessible by anyone.
    """
    if not thread_id:
        return
    owner = get_thread_owner(thread_id)
    if owner is not None and owner != user_id:
        raise HTTPException(
            status_code=403,
            detail="Access denied: this thread belongs to another user.",
        )


def generate_response(query: str, thread_id: str):
    """
    Generator function that streams the agent's output using the existing format logic.
    """
    if is_harmful(query):
        yield f"data: {json.dumps({'type': 'error', 'agent': 'system', 'content': 'I’m sorry, but I can’t share that.'})}\n\n"
        return

    answer_produced = False
    all_sources = []
    all_assets = []
    all_map = []
    all_stock = {}
    all_weather = {}
    seen_sources = set()
    seen_assets = set()

    try:
        config = {"configurable": {"thread_id": thread_id}}
        for content in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            subgraphs=True,
            stream_mode="updates",
            config=config,
        ):
            formatted = format_answer(content)

            if formatted:
                for item_str in formatted:
                    try:
                        item_obj = json.loads(item_str)
                        item_type = item_obj.get("type")

                        # 1. Accumulate metadata from the stream
                        if item_type == "sources":
                            for s in item_obj.get("sources", []):
                                key = (s.get("title"), s.get("url"), s.get("content"))
                                if key not in seen_sources:
                                    all_sources.append(s)
                                    seen_sources.add(key)

                        elif item_type == "assets":
                            for a in item_obj.get("assets", []):
                                if a not in seen_assets:
                                    all_assets.append(a)
                                    seen_assets.add(a)

                        elif item_type == "map":
                            all_map.extend(item_obj.get("map", []))

                        elif item_type == "stock":
                            all_stock.update(item_obj.get("stock", {}))

                        elif item_type == "weather":
                            all_weather.update(item_obj.get("weather", {}))

                        # 2. Intercept final answer and inject full collections
                        elif item_type == "answer":
                            answer_produced = True
                            try:
                                payload = json.loads(item_obj["content"])
                                if isinstance(payload, dict):
                                    payload["sources"] = all_sources
                                    payload["assets"] = all_assets
                                    if all_map:
                                        payload["map"] = all_map
                                    if all_stock:
                                        payload["stock"] = all_stock
                                    if all_weather:
                                        payload["weather"] = all_weather

                                    item_obj["content"] = json.dumps(
                                        payload, ensure_ascii=False
                                    )
                                    item_str = json.dumps(item_obj, ensure_ascii=False)
                            except Exception:
                                pass
                    except Exception:
                        pass

                    item_str, _ = _sanitize_stream_item(item_str)
                    yield f"data: {item_str}\n\n"

        if not answer_produced:
            print("Stream ended. Answer might escaped.")
            yield f"data: {json.dumps({'type': 'error', 'agent': 'system', 'content': 'Stream ended. Answer might escaped.'})}\n\n"

    except Exception as e:
        import traceback

        traceback.print_exc()
        # logger.error(f"Error during streaming: {e}")
        # Return an error message in a compatible format
        error_response = json.dumps(
            {"type": "error", "agent": "system", "content": str(e)}
        )
        yield f"data: {error_response}\n\n"


def _coerce_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    if isinstance(value, dict):
        return str(value.get("text", ""))
    return ""


def _sanitize_stream_item(item: str) -> tuple[str, bool]:
    try:
        payload = json.loads(item)
    except Exception:
        return item, False

    blocked = False
    safe_message = "I’m sorry, but I can’t share that."
    content = payload.get("content")
    event_type = payload.get("type")

    if isinstance(content, str):
        if event_type == "answer":
            try:
                answer_payload = json.loads(content)
            except Exception:
                answer_payload = None

            if isinstance(answer_payload, dict) and isinstance(
                answer_payload.get("answer"), str
            ):
                answer_text = answer_payload.get("answer", "")
                if has_prompt_leakage(answer_text):
                    answer_payload["answer"] = safe_message
                    payload["content"] = json.dumps(answer_payload, ensure_ascii=False)
                    blocked = True
            elif has_prompt_leakage(content):
                payload["content"] = safe_message
                blocked = True
        elif has_prompt_leakage(content):
            payload["content"] = safe_message
            blocked = True

    if blocked:
        payload["type"] = "error"
        payload["agent"] = "system"

    return json.dumps(payload, ensure_ascii=False), blocked


def _slice_messages_for_current_query(messages: list, current_query_text: str) -> list:
    if not messages or not current_query_text:
        return messages

    anchor_index = -1
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if getattr(msg, "type", None) != "human":
            continue
        if current_query_text in _coerce_text(getattr(msg, "content", "")):
            anchor_index = idx
            break

    if anchor_index < 0:
        return messages
    return messages[anchor_index + 1 :]


def _extract_light_answer(res: dict) -> str:
    messages = res.get("messages", []) if isinstance(res, dict) else []
    for msg in reversed(messages):
        if getattr(msg, "type", None) != "ai":
            continue
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(str(block.get("text", "")))
                elif isinstance(block, str):
                    text_parts.append(block)
            joined = "".join(text_parts).strip()
            if joined:
                return joined
    return ""


def _parse_tool_payload(content) -> dict | list | None:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        return content
    if not isinstance(content, str):
        return None

    text = content.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass

    return None


def _extract_light_metadata(
    res: dict,
    current_query_text: str = "",
) -> tuple[list[dict], list[dict], dict, dict]:
    messages = res.get("messages", []) if isinstance(res, dict) else []
    messages = _slice_messages_for_current_query(messages, current_query_text)
    sources: list[dict] = []
    map_results: list[dict] = []
    seen_sources: set[tuple[str, str, str]] = set()
    stock_payload: dict = {}
    weather_payload: dict = {}

    for msg in messages:
        if getattr(msg, "type", None) != "tool":
            continue

        tool_name = getattr(msg, "name", "") or ""
        payload = _parse_tool_payload(getattr(msg, "content", ""))
        if tool_name in ["google_search_light", "load_web_page_light"]:
            items = []
            if tool_name == "google_search_light":
                items = (
                    payload
                    if isinstance(payload, list)
                    else payload.get("results", [])
                    if isinstance(payload, dict)
                    else []
                )
            elif tool_name == "load_web_page_light":
                if isinstance(payload, dict):
                    items = [payload]

            for item in items:
                if not isinstance(item, dict):
                    continue

                content = str(item.get("content", "") or "").strip()
                if tool_name == "load_web_page_light" and len(content) > 100:
                    content = content[:100] + "..."

                source = {
                    "title": str(item.get("title", "") or "").strip(),
                    "url": str(item.get("url", "") or "").strip(),
                    "content": content,
                }
                key = (source["title"], source["url"], source["content"])
                if key in seen_sources or not any(key):
                    continue
                sources.append(source)
                seen_sources.add(key)

        if tool_name == "google_search_places_light":
            items = (
                payload
                if isinstance(payload, list)
                else payload.get("results", [])
                if isinstance(payload, dict)
                else []
            )
            for item in items:
                if not isinstance(item, dict):
                    continue
                map_results.append(item)

        if not isinstance(payload, dict):
            continue

        if tool_name == "get_stock_data_light":
            stock_candidate = payload.get("stock")
            if isinstance(stock_candidate, dict):
                stock_payload = stock_candidate

        if tool_name == "get_weather_light":
            weather_payload = payload

    return sources, map_results, stock_payload, weather_payload


@app.post("/chat")
async def chat(
    request: QueryRequest,
    user_id: str = Depends(get_current_user_with_rate_limit),
):
    """
    Endpoint to interact with the agent.
    Returns a streaming response of formatted JSON objects.
    Fire-and-forget: update threads_control.updated_at asynchronously.
    """
    _assert_thread_access(request.thread_id, user_id)
    if request.thread_id:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_db_executor, touch_thread, request.thread_id, user_id)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        generate_response(request.query, request.thread_id),
        media_type="text/event-stream",
        headers=headers,
    )


def light_generate_response(query_text: str, message: dict, config: dict):
    if is_harmful(query_text):
        yield f"data: {json.dumps({'type': 'error', 'agent': 'system', 'content': 'I’m sorry, but I can’t share that.'})}\n\n"
        return

    answer_produced = False

    try:
        seen_tools_emitted = set()
        all_sources = []
        all_map = []
        all_stock = {}
        all_weather = {}
        full_answer = ""

        for data in omni_light_agent.stream(
            message, config=config, stream_mode="updates"
        ):
            if True:  # Kept for indentation logic
                for node_name, node_output in data.items():
                    if isinstance(node_output, dict) and "messages" in node_output:
                        msgs = node_output["messages"]
                        if not isinstance(msgs, list):
                            msgs = [msgs]

                        # 1. Check if the model is calling a tool right now
                        for msg in msgs:
                            if getattr(msg, "type", None) == "ai":
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        tool_name = tc.get("name")
                                        args = tc.get("args", {})
                                        if tool_name:
                                            payload_str = json.dumps(
                                                {
                                                    "type": "tool_call",
                                                    "tool": tool_name,
                                                    "args": args,
                                                },
                                                ensure_ascii=False,
                                            )
                                            if payload_str not in seen_tools_emitted:
                                                yield f"data: {payload_str}\n\n"
                                                seen_tools_emitted.add(payload_str)

                                # 2. Check if the model is yielding a text answer
                                if getattr(msg, "content", None) and isinstance(
                                    msg.content, str
                                ):
                                    content_str = msg.content
                                    if has_prompt_leakage(content_str):
                                        content_str = (
                                            "I’m sorry, but I can’t share that."
                                        )
                                    # Streaming answer chunk
                                    # yield f"data: {json.dumps({'type': 'answer_chunk', 'answer': content_str}, ensure_ascii=False)}\n\n"
                                    full_answer += content_str
                                    answer_produced = True

                        # 3. Check if tools just ran and returned data
                        sources, map_results, stock, weather = _extract_light_metadata(
                            {"messages": msgs}, query_text
                        )
                        if sources:
                            all_sources.extend(sources)
                        if map_results:
                            all_map.extend(map_results)
                        if stock:
                            all_stock.update(stock)
                        if weather:
                            all_weather.update(weather)

                        payload = {}
                        if sources:
                            payload["source"] = sources
                            payload["sources"] = sources
                        if map_results:
                            payload["map"] = map_results
                        if stock:
                            payload["stock"] = stock
                        if weather:
                            payload["weather"] = weather

                        # tool_data is discarded as per patch request; data is accumulated for final answer payload
                        pass

        # Final unified payload to ensure no info is lost and keys match old format
        final_payload = {
            "type": "answer",
            "answer": full_answer,
            "sources": all_sources,
            "map": all_map,
            "stock": all_stock,
            "weather": all_weather,
        }
        yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"

        if not answer_produced and not full_answer:
            # Fallback for answer escaping
            pass

    except Exception as e:
        import traceback

        traceback.print_exc()
        yield f"data: {json.dumps({'type': 'error', 'agent': 'system', 'content': str(e)})}\n\n"


@app.post("/light_chat")
async def light_chat(
    request: QueryRequest,
    user_id: str = Depends(get_current_user),
):
    _assert_thread_access(request.thread_id, user_id)
    # Fire-and-forget: update threads_control.updated_at asynchronously
    if request.thread_id:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_db_executor, touch_thread, request.thread_id, user_id)

    config = {"configurable": {"thread_id": request.thread_id}}
    personalization = format_personalization(request.personalization)
    message_str = (
        f"User Query: {request.query}\n\nPersonalization: {personalization}\n\n"
    )
    if request.follow_up_content:
        message_str += f"\n\nFollow up text selection: {request.follow_up_content}"
    message = {
        "messages": [
            {
                "role": "user",
                "content": message_str,
            }
        ]
    }
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        light_generate_response(request.query, message, config),
        media_type="text/event-stream",
        headers=headers,
    )


@app.post("/research_helper")
async def research_helper(
    request: QueryRequest,
    user_id: str = Depends(get_current_user_check_rate_limit),
):
    if is_harmful(request.query):
        return json.dumps(
            {
                "response": "I'm sorry, but I can't share that.",
                "read_to_begin_research": False,
                "rewritten_query": "",
            }
        )
    _assert_thread_access(request.thread_id, user_id)
    # Fire-and-forget: update threads_control.updated_at asynchronously
    if request.thread_id:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_db_executor, touch_thread, request.thread_id, user_id)

    config = {"configurable": {"thread_id": request.thread_id}}
    personalization = format_personalization(request.personalization)
    message_str = (
        f"User Query: {request.query}\n\nPersonalization: {personalization}\n\n"
    )
    if request.follow_up_content:
        message_str += f"\n\nFollow up text selection: {request.follow_up_content}"
    message = {
        "messages": [
            {
                "role": "user",
                "content": message_str,
            }
        ]
    }
    res = omni_research_helper.invoke(message, config=config)
    to_return = {
        "response": res["structured_response"].response,
        "read_to_begin_research": res["structured_response"].read_to_begin_research,
        "rewritten_query": res["structured_response"].rewritten_query,
    }
    return to_return


@app.post("/check_source")
def check_source_api(request: CheckSourceRequest):
    try:
        source_data = request.source
        if len(request.text_selection) < 10:
            return {"error": "Text selection is too short"}
        sources = source_data.get("final_sources", [])
        if not sources:
            return {"error": "No sources found"}
        res = check_source(source_data, request.text_selection)
        return res
    except Exception as e:
        return {"error": str(e)}


@app.get("/get_thread_id")
def get_thread_id(user_id: str | None = Depends(get_optional_user)):
    """
    Generate a new thread ID and register it in both threads_control and user_threads.
    If auth headers are present the thread is immediately bound to the user.
    Guests are capped at GUEST_MAX_THREADS active threads.
    """
    if user_id and user_id.startswith("guest_"):
        if count_user_threads(user_id) >= GUEST_MAX_THREADS:
            raise HTTPException(
                status_code=429,
                detail=f"Guest accounts are limited to {GUEST_MAX_THREADS} threads. Please sign in for unlimited threads.",
            )
    thread_id = str(uuid.uuid4())
    upsert_thread(thread_id, user_id)
    if user_id:
        register_thread(thread_id, user_id)
    return thread_id


@app.post("/get_title")
def generate_title(request: QueryRequest):
    return get_title(request.query)


@app.post("/get_model")
def get_model(request: QueryRequest):
    res = get_auto_select_model(request.query)
    if res == "smart":
        res = "canvas"
    else:
        res = "light"
    # print(f"Query: {request.query}, Model: {res}")
    return json.dumps({"model": res})


@app.post("/update_memories")
async def update_memories_api(request: UpdateMemoriesRequest):
    # print(
    #     f"Past queries: {request.past_queries}, Past memories: {request.past_memories}"
    # )
    res = await get_update_memories(request.past_queries, request.past_memories)
    return res


@app.post("/api/sst")
async def speech_to_text_api(
    file: UploadFile = File(...),
):
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Only audio files are supported.")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        text = await get_text_from_audio(audio_bytes)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SST failed: {str(e)}")


@app.get("/health")
async def health():
    # Fire-and-forget: clean up stale threads asynchronously
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_db_executor, cleanup_old_threads)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# User threads – auth-gated endpoints
# ---------------------------------------------------------------------------


class SyncThreadRequest(BaseModel):
    messages: list
    title: str | None = None


class MergeRequest(BaseModel):
    guest_id: str


@app.get("/api/threads")
def api_get_threads(user_id: str = Depends(get_current_user)):
    """Return the list of threads owned by the current user."""
    threads = get_threads_for_user(user_id)
    # Serialise datetime objects so they become JSON-safe strings
    for t in threads:
        if hasattr(t.get("updated_at"), "isoformat"):
            t["updated_at"] = t["updated_at"].isoformat()
    return {"threads": threads}


@app.get("/api/threads/{thread_id}")
def api_get_thread(thread_id: str, user_id: str = Depends(get_current_user)):
    """Return the stored ui_messages for a single thread."""
    messages = get_thread_messages(thread_id, user_id)
    if messages is None:
        raise HTTPException(
            status_code=404, detail="Thread not found or access denied."
        )
    return {"messages": messages}


@app.post("/api/threads/{thread_id}/sync")
def api_sync_thread(
    thread_id: str,
    body: SyncThreadRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Upsert the ui_messages (and optionally the title) for a thread.
    The row is only written if it belongs to the requesting user.
    """
    ok = upsert_thread_messages(thread_id, user_id, body.messages)
    if not ok:
        raise HTTPException(
            status_code=404, detail="Thread not found or access denied."
        )
    if body.title:
        title_ok = update_thread_title(thread_id, user_id, body.title)
        if not title_ok:
            raise HTTPException(
                status_code=404, detail="Thread not found or access denied."
            )
    return {"status": "success"}


@app.get("/api/guests/daily-quota")
def api_guest_daily_quota(user_id: str = Depends(get_current_user)):
    """
    Return the remaining canvas-mode quota for a guest user today.
    Signed-in users always get unlimited (-1).
    """
    if not user_id.startswith("guest_"):
        return {"daily_limit": -1, "used": 0, "remaining": -1}
    used = _get_guest_usage_today(user_id)
    remaining = max(GUEST_DAILY_LIMIT - used, 0)
    return {"daily_limit": GUEST_DAILY_LIMIT, "used": used, "remaining": remaining}


@app.post("/api/users/merge")
def api_merge_guest(
    body: MergeRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Migrate all threads from a guest account to the authenticated user.
    Updates both user_threads (UI state) and threads_control (LangGraph state).
    Must be called with a valid Bearer token (i.e. after sign-in).
    """
    if not body.guest_id.startswith("guest_"):
        raise HTTPException(status_code=400, detail="Invalid guest_id format.")
    if user_id.startswith("guest_"):
        raise HTTPException(status_code=403, detail="Must be signed in to merge.")
    count = merge_guest_to_user(user_id, body.guest_id)
    # Mirror the reassignment in threads_control so retention rules apply correctly
    reassign_threads_user(body.guest_id, user_id)
    return {"status": "merged", "threads_migrated": count}


@app.delete("/api/threads/{thread_id}")
def api_delete_thread(
    thread_id: str,
    user_id: str = Depends(get_current_user),
):
    """
    Hard-delete a thread.
    Removes the UI record (user_threads) and the full LangGraph checkpoint state
    (checkpoints, checkpoint_blobs, checkpoint_writes, threads_control).
    Returns 404 if the thread doesn't belong to this user.
    """
    owned = delete_user_thread(thread_id, user_id)
    if not owned:
        raise HTTPException(
            status_code=404, detail="Thread not found or access denied."
        )
    # Clean up LangGraph state + threads_control row
    delete_thread_state(thread_id)
    return {"status": "deleted"}


class PatchTitleRequest(BaseModel):
    title: str


class PatchPinRequest(BaseModel):
    is_pinned: bool


@app.patch("/api/threads/{thread_id}/title")
def api_rename_thread(
    thread_id: str,
    body: PatchTitleRequest,
    user_id: str = Depends(get_current_user),
):
    """Rename a thread. Only the owning user can rename."""
    ok = update_thread_title(thread_id, user_id, body.title)
    if not ok:
        raise HTTPException(
            status_code=404, detail="Thread not found or access denied."
        )
    return {"status": "updated"}


@app.patch("/api/threads/{thread_id}/pin")
def api_pin_thread(
    thread_id: str,
    body: PatchPinRequest,
    user_id: str = Depends(get_current_user),
):
    """
    Pin or unpin a thread.
    Pinned threads are sorted to the top of the list and exempted from auto-cleanup.
    """
    ok = pin_user_thread(thread_id, user_id, body.is_pinned)
    if not ok:
        raise HTTPException(
            status_code=404, detail="Thread not found or access denied."
        )
    # Mirror pin state in threads_control so cleanup respects it
    pin_thread_state(thread_id, body.is_pinned)
    return {"status": "updated", "is_pinned": body.is_pinned}


# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)
