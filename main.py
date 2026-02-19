import dotenv

dotenv.load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import uuid
import logging
from core.light_agent import omni_light_agent

# Import the agent and formatter from existing codebase
from core.supervisor import agent
from core.utils.format import format_answer
from core.get_title import get_title
from core.auto_select_model import get_auto_select_model
from core.source_checker import check_source

app = FastAPI(title="Omni Agent API")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    thread_id: str = None
    follow_up_content: str = None


class CheckSourceRequest(BaseModel):
    source: dict  # Check if source is a dict
    text_selection: str


def generate_response(query: str, thread_id: str):
    """
    Generator function that streams the agent's output using the existing format logic.
    """
    start_researching_item = {
        "type": "reasoning",
        "agent": "System",
        "content": "Hang tight! It may take a few minutes to complete the research. Please don't close this page while I'm working on it.",
        "raw": {},
    }

    yield f"data: {json.dumps(start_researching_item)}\n\n"

    answer_produced = False

    try:
        config = {"configurable": {"thread_id": thread_id}}
        # Replicating the logic from main.py
        # stream_mode="updates" and subgraphs=True are critical parameters used in main.py
        stream = agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            subgraphs=True,
            stream_mode="updates",
            config=config,
        )
        for content in stream:
            # Replicating the formatting logic from main.py
            # 1. Convert content to string (as main.py does)
            content_str = str(content)

            # 2. Pass to format_answer
            formatted = format_answer(content_str)

            # 3. Handle the output similar to main.py's writing logic
            if formatted:
                if isinstance(formatted, list):
                    for item in formatted:
                        if "final_answer" in str(item):
                            answer_produced = True
                        yield f"data: {item}\n\n"
                else:
                    # Fallback for non-list returns, though format_answer type hint says list[str]
                    yield f"data: {formatted}\n\n"
        print("Stream finished")
        if not answer_produced:
            yield f"data: {json.dumps({'type': 'error', 'agent': 'system', 'content': 'No answer produced'})}\n\n"

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
def chat(request: QueryRequest):
    """
    Endpoint to interact with the agent.
    Returns a streaming response of formatted JSON objects.
    """
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        generate_response(request.query, request.thread_id),
        media_type="text/event-stream",
        headers=headers,
    )


@app.post("/light_chat")
def light_chat(request: QueryRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    message = {"messages": [{"role": "user", "content": request.query}]}
    if request.follow_up_content:
        message = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{request.query}\n\nFollow up text selection: {request.follow_up_content}",
                }
            ]
        }
    res = omni_light_agent.invoke(message, config=config)
    return json.dumps({"answer": res.get("messages")[-1].content})


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
    return str(uuid.uuid4())


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


@app.get("/health")
def health():
    return {"status": "ok"}


# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run(app, host="0.0.0.0", port=8000)
