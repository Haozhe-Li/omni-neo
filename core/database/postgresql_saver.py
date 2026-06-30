from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool, ConnectionPool
from psycopg.rows import dict_row
import os

DB_URI = os.getenv("DB_URI")

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": None,
    "row_factory": dict_row,
    # TCP keepalives so a connection killed by a middlebox/DB idle timeout is
    # detected quickly instead of surfacing as "SSL error: unexpected eof" mid-query.
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
}

# Async pool — used exclusively by AsyncPostgresSaver (LangGraph checkpointer).
# `check` validates a connection right before handing it out (psycopg_pool only
# does this on its periodic background sweep otherwise), so a connection the
# server/network already closed gets discarded and replaced instead of being
# handed to the checkpointer and failing with an SSL EOF.
pool = AsyncConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
    check=AsyncConnectionPool.check_connection,
    open=False,
)

# Sync pool — used by db_threads_control, db_user_threads, db_user_files.
# These modules run synchronous DB calls from FastAPI sync endpoints /
# thread-pool workers, so they cannot use the async pool.
sync_pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=10,
    kwargs=connection_kwargs,
    check=ConnectionPool.check_connection,
    open=False,
)

checkpointer = None


async def setup_checkpointer():
    global checkpointer
    sync_pool.open()
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()


async def teardown_checkpointer():
    await pool.close()
    sync_pool.close()
