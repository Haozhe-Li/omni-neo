import dotenv

dotenv.load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import logging

# Import the agent and formatter from existing codebase
from core.supervisor import agent
from core.utils.format import format_answer
from core.get_title import get_title

# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

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


def generate_response(query: str):
    """
    Generator function that streams the agent's output using the existing format logic.
    """
    # logger.info(f"Received query: {query}")
    # start_researching_item = {
    #     "type": "reasoning",
    #     "agent": "System",
    #     "content": f"Great! I'm working on your query now. Hang tight!",
    #     "raw": {},
    # }

    # yield f"data: {json.dumps(start_researching_item)}\n\n"

    try:
        # Replicating the logic from main.py
        # stream_mode="updates" and subgraphs=True are critical parameters used in main.py
        stream = agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            subgraphs=True,
            stream_mode="updates",
            # config={"recursion_limit": 20}, # Optional config found in main.py comments
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
                        # item is expected to be a JSON string from format_answer
                        # We yield it followed by a newline for streaming
                        yield f"data: {item}\n\n"
                else:
                    # Fallback for non-list returns, though format_answer type hint says list[str]
                    yield f"data: {formatted}\n\n"

    except Exception as e:
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
        generate_response(request.query),
        media_type="text/event-stream",
        headers=headers,
    )


@app.post("/get_title")
def get_title(request: QueryRequest):
    return get_title(request.query)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
