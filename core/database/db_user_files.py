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
        is_rag_indexed BOOLEAN DEFAULT FALSE,

        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_user_files_thread_id ON user_files(thread_id);
    CREATE INDEX IF NOT EXISTS idx_user_files_user_id ON user_files(user_id);
"""

import logging
from core.database.postgresql_saver import sync_pool as pool
from core.utils.redis_cache import l1cache

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
            is_rag_indexed BOOLEAN DEFAULT FALSE,
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


def update_file_ready(
    file_id: str, extracted_text: str | None = None, is_rag_indexed: bool = False
) -> bool:
    """Update file status to 'ready' after parsing/indexing."""
    sql = """
        UPDATE user_files 
        SET status = 'ready',
            extracted_text = %s,
            is_rag_indexed = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE file_id = %s
    """
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (extracted_text, is_rag_indexed, file_id))
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


@l1cache(ttl=3600 * 3)
def get_thread_files(thread_id: str) -> list[dict]:
    """Fetch all files associated with a thread."""
    sql = "SELECT * FROM user_files WHERE thread_id = %s AND status = 'ready'"
    try:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (thread_id,))
                rows = cur.fetchall()
                return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"[db_user_files] get_thread_files error: {e}")
        return []
