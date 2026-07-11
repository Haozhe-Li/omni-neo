import os
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.auth import get_current_user
from core.database.db_user_files import create_pending_file, get_file_record
from core.RAG.file_parser import get_put_presigned_url, process_uploaded_file, DOCX_MIME_TYPE

router = APIRouter(prefix="/api/upload", tags=["uploads"])


class UploadUrlRequest(BaseModel):
    filename: str
    file_type: str
    file_size_bytes: int
    thread_id: str | None = None


@router.post("/url")
def api_upload_url(
    request: UploadUrlRequest,
    user_id: str = Depends(get_current_user),
):
    # Use thread_id from request body; generate one only if frontend didn't provide it.
    thread_id = request.thread_id or str(uuid.uuid4())
    raw_file_id = str(uuid.uuid4())
    file_id = f"user_uploads/{user_id}/{raw_file_id}"
    s3_bucket = os.getenv("S3_BUCKET_NAME", "omni")

    if request.file_type.startswith("image/"):
        category = "image"
    elif (
        request.file_type == "application/pdf"
        or request.file_type == DOCX_MIME_TYPE
        or request.file_type.startswith("text/")
        or request.file_type
        in [
            "application/json",
            "application/xml",
            "application/javascript",
            "application/x-javascript",
            "application/x-python",
            "application/x-sh",
            "application/x-httpd-php",
            "application/yaml",
            "application/x-yaml",
        ]
    ):
        category = "document"
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    create_pending_file(
        file_id=file_id,
        user_id=user_id,
        thread_id=thread_id,
        original_filename=request.filename,
        file_type=request.file_type,
        file_size_bytes=request.file_size_bytes,
        s3_bucket=s3_bucket,
        category=category,
    )

    url = get_put_presigned_url(s3_bucket, file_id, request.file_type)
    return {"upload_url": url, "file_id": file_id, "thread_id": thread_id}


@router.post("/confirm")
def api_upload_confirm(file_id: str, user_id: str = Depends(get_current_user)):
    # Sync route — FastAPI/Starlette already runs this off the event loop in
    # its own thread pool, so blocking here doesn't stall other requests.
    # Blocking (not fire-and-forget) so the caller only gets a response once
    # the file is actually ready/failed — no more racing a later /chat call
    # against parsing that hasn't finished yet.
    process_uploaded_file(file_id)
    record = get_file_record(file_id)
    if not record:
        raise HTTPException(status_code=404, detail="File not found.")
    if record["status"] == "failed":
        # Non-2xx so a normal fetch/axios caller can't mistake this for success
        # by only checking the HTTP status and ignoring the response body.
        raise HTTPException(status_code=422, detail="File processing failed.")
    return {"status": record["status"], "file_id": file_id}
