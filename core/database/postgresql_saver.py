from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
import os

DB_URI = os.getenv("DB_URI")

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": None,
    "row_factory": dict_row,
}

pool = AsyncConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
    open=False,
)

checkpointer = None


async def setup_checkpointer():
    global checkpointer
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()


async def teardown_checkpointer():
    await pool.close()
