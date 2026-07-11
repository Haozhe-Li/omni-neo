"""
Database operations for the user_files table.

Table schema:
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
from core.database.postgresql_saver import sync_pool as pool

logger = logging.getLogger(__name__)


def setup_user_files_table() -> None:
    """Create the user_files table and indexes if they don't exist."""
    sql_table = """
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
    """
    sql_index1 = (
        "CREATE INDEX IF NOT EXISTS idx_user_files_thread_id ON user_files(thread_id);"
    )
    sql_index2 = (
        "CREATE INDEX IF NOT EXISTS idx_user_files_user_id ON user_files(user_id);"
    )

    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql_table)
                cur.execute(sql_index1)
                cur.execute(sql_index2)
    except Exception as e:
        logger.error(f"[db_user_files] setup_user_files_table error: {e}")


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
    sql = """
        INSERT INTO user_files (
            file_id, user_id, thread_id, original_filename, file_type, 
            file_size_bytes, s3_bucket, category, status
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending')
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        file_id,
                        user_id,
                        thread_id,
                        original_filename,
                        file_type,
                        file_size_bytes,
                        s3_bucket,
                        category,
                    ),
                )
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"[db_user_files] create_pending_file error: {e}")
        return False


def get_file_record(file_id: str) -> dict | None:
    """Fetch a file record by file_id."""
    sql = "SELECT * FROM user_files WHERE file_id = %s"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (file_id,))
                row = cur.fetchone()
                return dict(row) if row else None
    except Exception as e:
        logger.error(f"[db_user_files] get_file_record error: {e}")
        return None


def update_file_ready(file_id: str, extracted_text: str | None = None) -> bool:
    """Update file status to 'ready' after parsing."""
    sql = """
        UPDATE user_files
        SET status = 'ready',
            extracted_text = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE file_id = %s
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (extracted_text, file_id))
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"[db_user_files] update_file_ready error: {e}")
        return False


def update_file_failed(file_id: str) -> bool:
    """Update file status to 'failed'."""
    sql = "UPDATE user_files SET status = 'failed', updated_at = CURRENT_TIMESTAMP WHERE file_id = %s"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (file_id,))
                return cur.rowcount > 0
    except Exception as e:
        logger.error(f"[db_user_files] update_file_failed error: {e}")
        return False


def get_user_file_buckets(user_id: str) -> list[str]:
    """Return the distinct S3 buckets this user's files live in (usually just one)."""
    sql = "SELECT DISTINCT s3_bucket FROM user_files WHERE user_id = %s AND s3_bucket IS NOT NULL"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return [r["s3_bucket"] for r in cur.fetchall()]
    except Exception as e:
        logger.error(f"[db_user_files] get_user_file_buckets error: {e}")
        return []


def delete_user_files(user_id: str) -> int:
    """Delete every user_files row for this user. Returns the number of rows deleted.

    Caller is responsible for also removing the underlying S3 objects
    (see delete_user_uploads_from_s3 in core/RAG/file_parser.py).
    """
    sql = "DELETE FROM user_files WHERE user_id = %s"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                return cur.rowcount
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
    sql = """
        SELECT COUNT(*) AS count FROM user_files
        WHERE thread_id = %s AND original_filename = %s AND status = 'ready'
          AND (created_at, file_id) < (%s, %s)
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (thread_id, filename, created_at, file_id))
                row = cur.fetchone()
                return row["count"] if row else 0
    except Exception as e:
        logger.error(f"[db_user_files] count_prior_ready_files_with_name error: {e}")
        return 0
