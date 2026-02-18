from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
import os

DB_URI = os.getenv("DB_URI")

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": None,
    "row_factory": dict_row,
}

pool = ConnectionPool(
    conninfo=DB_URI,
    max_size=20,
    kwargs=connection_kwargs,
)

checkpointer = PostgresSaver(pool)
checkpointer.setup()
