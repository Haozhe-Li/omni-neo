"""
Database operations for the user_files table.

All access is over Supabase's PostgREST HTTP API (see supabase_client.py).

Table schema (managed in Supabase, see schema.sql):
    CREATE TABLE IF NOT EXISTS user_files (
        file_id VARCHAR(255) PRIMARY KEY,
        user_id VARCHAR(255) NOT NULL,
        thread_id VARCHAR(255) NOT NULL,

        original_filename VARCHAR(255) NOT NULL,
        file_type VARCHAR(255) NOT NULL,
        file_size_bytes BIGINT DEFAULT 0,

        status VARCHAR(50) DEFAULT 'pending',
        s3_bucket VARCHAR(255),

        category VARCHAR(50) NOT NULL,

        extracted_text TEXT,

        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_user_files_thread_id ON user_files(thread_id);
    CREATE INDEX IF NOT EXISTS idx_user_files_user_id ON user_files(user_id);
"""

import logging

from core.database.supabase_client import supabase, utcnow_iso

logger = logging.getLogger(__name__)


def create_pending_file(
    file_id: str,
    user_id: str,
    thread_id: str,
    original_filename: str,
    file_type: str,
    file_size_bytes: int,
    s3_bucket: str,
    category: str,
) -> bool:
    """Insert a new file record with 'pending' status."""
    try:
        res = supabase.table("user_files").insert({
            "file_id": file_id,
            "user_id": user_id,
            "thread_id": thread_id,
            "original_filename": original_filename,
            "file_type": file_type,
            "file_size_bytes": file_size_bytes,
            "s3_bucket": s3_bucket,
            "category": category,
            "status": "pending",
        }).execute()
        return bool(res.data)
    except Exception as e:
        logger.error(f"[db_user_files] create_pending_file error: {e}")
        return False


def get_file_record(file_id: str) -> dict | None:
    """Fetch a file record by file_id."""
    try:
        res = (
            supabase.table("user_files")
            .select("*")
            .eq("file_id", file_id)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None
    except Exception as e:
        logger.error(f"[db_user_files] get_file_record error: {e}")
        return None


def update_file_ready(file_id: str, extracted_text: str | None = None) -> bool:
    """Update file status to 'ready' after parsing."""
    try:
        res = supabase.table("user_files").update({
            "status": "ready",
            "extracted_text": extracted_text,
            "updated_at": utcnow_iso(),
        }).eq("file_id", file_id).execute()
        return bool(res.data)
    except Exception as e:
        logger.error(f"[db_user_files] update_file_ready error: {e}")
        return False


def update_file_failed(file_id: str) -> bool:
    """Update file status to 'failed'."""
    try:
        res = supabase.table("user_files").update({
            "status": "failed",
            "updated_at": utcnow_iso(),
        }).eq("file_id", file_id).execute()
        return bool(res.data)
    except Exception as e:
        logger.error(f"[db_user_files] update_file_failed error: {e}")
        return False


def get_user_file_buckets(user_id: str) -> list[str]:
    """Return the distinct S3 buckets this user's files live in (usually just one)."""
    try:
        res = (
            supabase.table("user_files")
            .select("s3_bucket")
            .eq("user_id", user_id)
            .not_.is_("s3_bucket", "null")
            .execute()
        )
        return list({r["s3_bucket"] for r in res.data if r.get("s3_bucket")})
    except Exception as e:
        logger.error(f"[db_user_files] get_user_file_buckets error: {e}")
        return []


def delete_user_files(user_id: str) -> int:
    """Delete every user_files row for this user. Returns the number of rows deleted.

    Caller is responsible for also removing the underlying S3 objects
    (see delete_user_uploads_from_s3 in core/RAG/file_parser.py).
    """
    try:
        res = supabase.table("user_files").delete().eq("user_id", user_id).execute()
        return len(res.data or [])
    except Exception as e:
        logger.error(f"[db_user_files] delete_user_files error: {e}")
        return 0


def count_prior_ready_files_with_name(thread_id: str, filename: str, file_id: str, created_at) -> int:
    """Count ready files in a thread sharing `filename`, ordered strictly before
    (`created_at`, `file_id`).

    Used to assign Finder-style suffixes (name.ext, name(1).ext, name(2).ext, ...)
    when mounting documents into the agent's virtual filesystem, so re-uploads of
    a same-named file don't collide. Ordering by (created_at, file_id) rather than
    created_at alone gives a strict total order even when two files share a
    timestamp, so files in the same upload batch don't double-count each other.

    PostgREST can't express the SQL row-tuple comparison ``(created_at, file_id)
    < (X, Y)``, so the same-name ready set (small — one thread, one filename) is
    fetched and the strict-order count is computed here.
    """
    try:
        res = (
            supabase.table("user_files")
            .select("file_id, created_at")
            .eq("thread_id", thread_id)
            .eq("original_filename", filename)
            .eq("status", "ready")
            .execute()
        )
        pivot = (str(created_at), file_id)
        return sum(
            1 for r in (res.data or [])
            if (str(r["created_at"]), r["file_id"]) < pivot
        )
    except Exception as e:
        logger.error(f"[db_user_files] count_prior_ready_files_with_name error: {e}")
        return 0
