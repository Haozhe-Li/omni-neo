import asyncio

from fastapi import APIRouter, HTTPException, UploadFile, File

from core.utils.data_model import QueryRequest, AutoCompleteRequest
from core.auto_complete import auto_complete
from core.get_title import get_title
from core.audio_sst import get_text_from_audio
from core.database.db_threads_control import cleanup_old_threads
from core.routers.state import db_executor

router = APIRouter(tags=["misc"])


@router.post("/auto_complete")
async def api_auto_complete(request: AutoCompleteRequest):
    """Endpoint for text autocomplete."""
    try:
        results = auto_complete(request.text.strip())
        return {"texts": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get_title")
def generate_title(request: QueryRequest):
    return get_title(request.query)


@router.post("/api/sst")
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


@router.get("/health")
async def health():
    # Fire-and-forget: clean up stale threads asynchronously
    loop = asyncio.get_event_loop()
    loop.run_in_executor(db_executor, cleanup_old_threads)
    return {"status": "ok"}
