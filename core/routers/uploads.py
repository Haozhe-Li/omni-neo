import os
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.auth import get_current_user
from core.database.db_user_files import create_pending_file, get_file_record
from core.RAG.file_parser import get_put_presigned_url, process_uploaded_file, MARKDOWN_SOURCE_EXTENSIONS

router = APIRouter(prefix="/api/upload", tags=["uploads"])

# Plain-text/code formats accepted alongside the anydoc-converted document
# formats — mirrors the frontend's own allow-list (search-home.tsx) so a file
# that clears the browser's picker doesn't get rejected here. Checked by
# extension OR MIME (like MARKDOWN_SOURCE_EXTENSIONS below): browsers report
# unreliable/generic MIME types for a lot of these too.
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".py", ".js", ".jsx", ".ts", ".tsx", ".html",
    ".json", ".xml", ".yaml", ".yml", ".java", ".c", ".cpp", ".h", ".hpp", ".sh",
}
_TEXT_MIME_TYPES = {
    "application/json",
    "application/xml",
    "application/javascript",
    "application/x-javascript",
    "application/x-python",
    "application/x-sh",
    "application/x-httpd-php",
    "application/yaml",
    "application/x-yaml",
}


class UploadUrlRequest(BaseModel):
    filename: str
    file_type: str
    file_size_bytes: int
    thread_id: str | None = None


def classify_upload(filename: str, file_type: str) -> str:
    """'image' or 'document' for an upload, or 400 if the format is not accepted.

    One rule for every caller (this router and the training-data collector), so a
    file is filed under the same category whichever door it came through.
    """
    ext = os.path.splitext(filename)[1].lower()
    if file_type.startswith("image/"):
        return "image"
    if (
        ext in MARKDOWN_SOURCE_EXTENSIONS
        or ext in _TEXT_EXTENSIONS
        or file_type.startswith("text/")
        or file_type in _TEXT_MIME_TYPES
    ):
        return "document"
    raise HTTPException(status_code=400, detail="Unsupported file format")


def mint_upload(
    *, user_id: str, thread_id: str | None, filename: str, file_type: str, file_size_bytes: int
) -> dict:
    """Register a pending file and return its presigned S3 PUT URL.

    The whole of POST /api/upload/url, so the collector (core/routers/collector.py)
    stores files exactly as a user's upload does: same key shape
    (`user_uploads/<user>/<uuid>`), same bucket, same `user_files` row, same category.
    """
    # Use thread_id from request body; generate one only if frontend didn't provide it.
    thread_id = thread_id or str(uuid.uuid4())
    raw_file_id = str(uuid.uuid4())
    file_id = f"user_uploads/{user_id}/{raw_file_id}"
    s3_bucket = os.getenv("S3_BUCKET_NAME", "omni")

    category = classify_upload(filename, file_type)

    create_pending_file(
        file_id=file_id,
        user_id=user_id,
        thread_id=thread_id,
        original_filename=filename,
        file_type=file_type,
        file_size_bytes=file_size_bytes,
        s3_bucket=s3_bucket,
        category=category,
    )

    url = get_put_presigned_url(s3_bucket, file_id, file_type)
    return {"upload_url": url, "file_id": file_id, "thread_id": thread_id}


@router.post("/url")
def api_upload_url(
    request: UploadUrlRequest,
    user_id: str = Depends(get_current_user),
):
    return mint_upload(
        user_id=user_id,
        thread_id=request.thread_id,
        filename=request.filename,
        file_type=request.file_type,
        file_size_bytes=request.file_size_bytes,
    )


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
