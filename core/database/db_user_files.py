"""
Database operations for the user_files table (direct Postgres, see pg.py).

Table schema: see schema.sql.
"""

import logging

from core.database import pg

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
        pg.execute(
            "INSERT INTO user_files (file_id, user_id, thread_id, original_filename, file_type, "
            "file_size_bytes, s3_bucket, category, status) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending')",
            (file_id, user_id, thread_id, original_filename, file_type, file_size_bytes, s3_bucket, category),
        )
        return True
    except Exception as e:
        logger.error(f"[db_user_files] create_pending_file error: {e}")
        return False


def get_file_record(file_id: str) -> dict | None:
    """Fetch a file record by file_id."""
    try:
        return pg.fetch_one("SELECT * FROM user_files WHERE file_id = %s", (file_id,))
    except Exception as e:
        logger.error(f"[db_user_files] get_file_record error: {e}")
        return None


def update_file_ready(file_id: str, extracted_text: str | None = None) -> bool:
    """Update file status to 'ready' after parsing."""
    try:
        return pg.execute(
            "UPDATE user_files SET status = 'ready', extracted_text = %s, updated_at = now() "
            "WHERE file_id = %s",
            (extracted_text, file_id),
        ) > 0
    except Exception as e:
        logger.error(f"[db_user_files] update_file_ready error: {e}")
        return False


def update_file_failed(file_id: str) -> bool:
    """Update file status to 'failed'."""
    try:
        return pg.execute(
            "UPDATE user_files SET status = 'failed', updated_at = now() WHERE file_id = %s",
            (file_id,),
        ) > 0
    except Exception as e:
        logger.error(f"[db_user_files] update_file_failed error: {e}")
        return False


def get_user_file_buckets(user_id: str) -> list[str]:
    """Return the distinct S3 buckets this user's files live in (usually just one)."""
    try:
        rows = pg.fetch_all(
            "SELECT DISTINCT s3_bucket FROM user_files WHERE user_id = %s AND s3_bucket IS NOT NULL",
            (user_id,),
        )
        return [r["s3_bucket"] for r in rows]
    except Exception as e:
        logger.error(f"[db_user_files] get_user_file_buckets error: {e}")
        return []


def delete_user_files(user_id: str) -> int:
    """Delete every user_files row for this user. Returns the number of rows deleted.

    Caller is responsible for also removing the underlying S3 objects
    (see delete_user_uploads_from_s3 in core/RAG/file_parser.py).
    """
    try:
        return pg.execute("DELETE FROM user_files WHERE user_id = %s", (user_id,))
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
    """
    try:
        row = pg.fetch_one(
            "SELECT count(*) AS n FROM user_files "
            "WHERE thread_id = %s AND original_filename = %s AND status = 'ready' "
            "AND (created_at, file_id) < (%s::timestamptz, %s)",
            (thread_id, filename, str(created_at), file_id),
        )
        return int(row["n"])
    except Exception as e:
        logger.error(f"[db_user_files] count_prior_ready_files_with_name error: {e}")
        return 0
