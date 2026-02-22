import dotenv

dotenv.load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
from core.light_agent import omni_light_agent
from core.database.db_threads_control import (
    upsert_thread,
    touch_thread,
    cleanup_old_threads,
)

# Import the agent and formatter from existing codebase
from core.supervisor import agent
from core.utils.format import format_answer
from core.get_title import get_title
from core.auto_select_model import get_auto_select_model
from core.source_checker import check_source
from core.query_rewriter import rewrite_query
from core.prompt_guard import is_harmful
from core.utils.data_model import (
    QueryRequest,
    CheckSourceRequest,
    Personalization,
    UpdateMemoriesRequest,
)
from core.utils.utils import format_personalization
from core.memories_update_llm import get_update_memories

app = FastAPI(title="Omni Agent API")

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


def generate_response(query: str, thread_id: str, personalization: str):
    """
    Generator function that streams the agent's output using the existing format logic.
    """
    # yield f"data: {json.dumps({'type': 'error', 'agent': 'system', 'content': 'Stream ended. Answer might escaped.'})}\n\n"
    # return

    if is_harmful(query):
        yield f"data: {json.dumps({'type': 'error', 'agent': 'system', 'content': 'I’m sorry, but I can’t share that.'})}\n\n"
        return

    start_researching_message = {
        "type": "tool",
        "tool": "write_todos",
        "agent": "Supervisor",
        "content": "Tool Calling",
        "raw": {
            "args": {
                "todos": [
                    {"content": "Understand the user's query", "status": "pending"},
                ]
            },
            "id": "call_42W2889jFBqKSKhfWQptY7An",
        },
    }

    yield f"data: {json.dumps(start_researching_message)}\n\n"

    rewritten_query = rewrite_query(query, personalization)

    finish_rewriting_message = {
        "type": "tool",
        "tool": "write_todos",
        "agent": "Supervisor",
        "content": "Tool Calling",
        "raw": {
            "args": {
                "todos": [
                    {"content": "Understand the user's query", "status": "completed"},
                ]
            },
            "id": "call_42W2889jFBqKSKhfWQptY7An",
        },
    }

    yield f"data: {json.dumps(finish_rewriting_message)}\n\n"

    show_rewritten_query_message = {
        "type": "reasoning",
        "agent": "Supervisor",
        "content": f"I've rewritten user's query to: {rewritten_query} Now let's start the research.",
        "raw": {},
    }

    yield f"data: {json.dumps(show_rewritten_query_message)}\n\n"

    answer_produced = False

    try:
        config = {"configurable": {"thread_id": thread_id}}
        # Replicating the logic from main.py
        # stream_mode="updates" and subgraphs=True are critical parameters used in main.py
        for content in agent.stream(
            {"messages": [{"role": "user", "content": rewritten_query}]},
            subgraphs=True,
            stream_mode="updates",
            config=config,
        ):
            # Pass full payload to format_answer (it will natively extract SupervisorOutputs and struct JSONs now)
            formatted = format_answer(content)

            # Handle the output similar to main.py's writing logic
            if formatted:
                if isinstance(formatted, list):
                    for item in formatted:
                        if 'type":"answer' in str(item).replace(
                            " ", ""
                        ) or '"type": "answer"' in str(item):
                            answer_produced = True
                        yield f"data: {item}\n\n"
                else:
                    # Fallback for non-list returns, though format_answer type hint says list[str]
                    yield f"data: {formatted}\n\n"

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


@app.post("/chat")
async def chat(request: QueryRequest):
    """
    Endpoint to interact with the agent.
    Returns a streaming response of formatted JSON objects.
    Fire-and-forget: update threads_control.updated_at asynchronously.
    """
    if request.thread_id:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_db_executor, touch_thread, request.thread_id)

    personalization = format_personalization(request.personalization)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        generate_response(request.query, request.thread_id, personalization),
        media_type="text/event-stream",
        headers=headers,
    )


@app.post("/light_chat")
async def light_chat(request: QueryRequest):
    if is_harmful(request.query):
        return json.dumps({"answer": "I’m sorry, but I can’t share that."})
    # Fire-and-forget: update threads_control.updated_at asynchronously
    if request.thread_id:
        loop = asyncio.get_event_loop()
        loop.run_in_executor(_db_executor, touch_thread, request.thread_id)

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
    res = omni_light_agent.invoke(message, config=config)
    return json.dumps(
        {
            "answer": res["structured_response"].answer,
            "use_search": res["structured_response"].use_search,
        }
    )


@app.post("/check_source")
def check_source_api(request: CheckSourceRequest):
    try:
        source_data = request.source
        if len(request.text_selection) < 10:
            return {"error": "Text selection is too short"}
        final_sources = source_data.get("final_sources", [])
        if not final_sources:
            return {"error": "No sources found"}
        res = check_source(source_data, request.text_selection)
        return res
    except Exception as e:
        return {"error": str(e)}


@app.get("/get_thread_id")
def get_thread_id():
    """
    Generate a new thread ID and synchronously register it in threads_control.
    This is the only blocking DB write in the thread lifecycle.
    """
    thread_id = str(uuid.uuid4())
    upsert_thread(thread_id)
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


@app.get("/health")
async def health():
    # Fire-and-forget: clean up stale threads asynchronously
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_db_executor, cleanup_old_threads)
    return {"status": "ok"}


# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)
