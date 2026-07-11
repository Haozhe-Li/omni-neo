import os
import boto3
import base64
import tempfile
import logging
import pymupdf
from docx import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from core.database.db_user_files import (
    update_file_ready,
    get_file_record,
    update_file_failed,
)

logger = logging.getLogger(__name__)

DOCX_MIME_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

# Formats we parse into text before mounting — `read_file` in the agent's
# filesystem only understands text, so binary formats must never be handed
# to it as-is; the raw file is only ever used to produce `extracted_text`.
MARKDOWN_SOURCE_TYPES = {"application/pdf", DOCX_MIME_TYPE}

# Initialize S3 Client
s3_client = boto3.client(
    "s3",
    endpoint_url=os.getenv("S3_ENDPOINT_URL"),
    aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
    region_name="auto",  # Often needed for R2/Custom endpoints
)


def _download_from_s3(bucket: str, key: str, local_path: str):
    s3_client.download_file(bucket, key, local_path)


def _iter_docx_blocks(document: DocxDocument):
    """Yield paragraphs and tables in document order (python-docx exposes them
    as separate lists, which would put every table at the end)."""
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            yield DocxParagraph(child, document)
        elif child.tag.endswith("}tbl"):
            yield DocxTable(child, document)


def _docx_paragraph_to_markdown(paragraph: DocxParagraph) -> str:
    text = paragraph.text.strip()
    if not text:
        return ""
    style = (paragraph.style.name or "").lower()
    if style.startswith("heading"):
        try:
            level = int(style.replace("heading", "").strip())
        except ValueError:
            level = 1
        return f"{'#' * max(1, min(level, 6))} {text}"
    if style.startswith("list bullet"):
        return f"- {text}"
    if style.startswith("list number"):
        return f"1. {text}"
    return text


def _docx_table_to_markdown(table: DocxTable) -> str:
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join("---" for _ in rows[0]) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return "\n".join(lines)


def _parse_docx_to_markdown(local_path: str) -> str:
    document = DocxDocument(local_path)
    parts = []
    for block in _iter_docx_blocks(document):
        md = (
            _docx_paragraph_to_markdown(block)
            if isinstance(block, DocxParagraph)
            else _docx_table_to_markdown(block)
        )
        if md:
            parts.append(md)
    return "\n\n".join(parts)


def process_uploaded_file(file_id: str):
    record = get_file_record(file_id)
    if not record:
        logger.error(f"Record {file_id} not found.")
        return

    if record["category"] == "image":
        update_file_ready(file_id)
        return

    # Download file for parsing
    with tempfile.TemporaryDirectory() as temp_dir:
        local_path = os.path.join(temp_dir, record["original_filename"])
        try:
            _download_from_s3(record["s3_bucket"], file_id, local_path)
        except Exception as e:
            logger.error(f"S3 download failed for {file_id}: {e}")
            update_file_failed(file_id)
            return

        try:
            full_text = ""
            if record["file_type"] == "application/pdf":
                with pymupdf.open(local_path) as pdf_doc:
                    full_text = "\n".join(page.get_text() for page in pdf_doc)
            elif record["file_type"] == DOCX_MIME_TYPE:
                full_text = _parse_docx_to_markdown(local_path)
            else:
                with open(local_path, "r", encoding="utf-8") as f:
                    full_text = f.read()

            update_file_ready(file_id, extracted_text=full_text)
            logger.info(f"File {file_id} processed ({len(full_text)} chars).")

        except Exception as e:
            logger.error(f"Parsing failed for {file_id}: {e}")
            update_file_failed(file_id)


def get_image_base64_data_url(file_id: str) -> str | None:
    """Download an image from S3 and inline it as a base64 data URI.

    Vision models get the bytes directly rather than a remote (presigned) URL —
    no dependency on the model provider being able to fetch an external link.
    """
    record = get_file_record(file_id)
    if not record or not record.get("s3_bucket"):
        return None
    try:
        obj = s3_client.get_object(Bucket=record["s3_bucket"], Key=file_id)
        data = obj["Body"].read()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{record['file_type']};base64,{encoded}"
    except Exception as e:
        logger.error(f"Download/encode image failed for {file_id}: {e}")
        return None


def delete_user_uploads_from_s3(user_id: str, buckets: list[str]) -> int:
    """Delete every object under the `user_uploads/{user_id}/` prefix in each
    given bucket. File keys are minted as `user_uploads/{user_id}/{uuid}`
    (see uploads.py), so this prefix alone accounts for every upload —
    including any orphaned objects whose user_files row is already gone.

    Best-effort: a bucket that errors out is logged and skipped rather than
    aborting the whole purge. Returns the number of objects deleted.
    """
    prefix = f"user_uploads/{user_id}/"
    deleted = 0
    for bucket in buckets:
        try:
            continuation_token = None
            while True:
                kwargs = {"Bucket": bucket, "Prefix": prefix}
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token
                resp = s3_client.list_objects_v2(**kwargs)
                contents = resp.get("Contents", [])
                if contents:
                    keys = [{"Key": obj["Key"]} for obj in contents]
                    s3_client.delete_objects(Bucket=bucket, Delete={"Objects": keys})
                    deleted += len(keys)
                if resp.get("IsTruncated"):
                    continuation_token = resp.get("NextContinuationToken")
                else:
                    break
        except Exception as e:
            logger.error(f"delete_user_uploads_from_s3 failed for bucket {bucket}: {e}")
    return deleted


def get_put_presigned_url(s3_bucket: str, file_id: str, file_type: str) -> str | None:
    """Generate a short-lived presigned URL for frontend direct upload"""
    try:
        url = s3_client.generate_presigned_url(
            "put_object",
            Params={"Bucket": s3_bucket, "Key": file_id, "ContentType": file_type},
            ExpiresIn=600,
        )
        return url
    except Exception as e:
        logger.error(f"Generate put presigned URL failed: {e}")
        return None
