"""Shared Postgres access: psycopg 3 over TCP, one pool per process for sync
callers and one for async callers.

Connection string comes from ``DATABASE_URL`` (on Railway: the service's
``postgres.railway.internal`` URL, so traffic never leaves the private network).

Why two pools. FastAPI runs ``def`` endpoints in a threadpool, where the sync pool
is the right tool; ``async def`` handlers (the chat hot path) must not block the
event loop for a round trip, so they use the async pool. Each is lazily created on
first use and closed by ``close_pools()`` at shutdown.

Every connection is autocommit: a lone statement is its own transaction, and
multi-statement units of work say so explicitly with ``with conn.transaction():``.

Rows come back as plain dicts with a few values normalised so the rest of the code
(and the JSON caches in front of it) sees simple types: timestamps and dates as ISO
strings, NUMERIC as float, UUID as str. JSONB columns arrive already parsed.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import decimal
import os
import threading
import uuid
from typing import Any, Iterable, Sequence

from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool, ConnectionPool

# ── row normalisation ───────────────────────────────────────────────────────


def _norm(value: Any) -> Any:
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def _row_factory(cursor):
    names = [d.name for d in cursor.description] if cursor.description else []

    def make(values: Sequence[Any]) -> dict:
        return {n: _norm(v) for n, v in zip(names, values)}

    return make


def adapt(value: Any) -> Any:
    """Python value -> SQL parameter. dict/list become JSONB; everything else is
    handled by psycopg's own adapters."""
    if isinstance(value, (dict, list)):
        return Jsonb(value)
    return value


# ── pools ───────────────────────────────────────────────────────────────────

_SYNC_MAX = int(os.getenv("PG_POOL_SYNC_MAX", "20"))
_ASYNC_MAX = int(os.getenv("PG_POOL_ASYNC_MAX", "10"))
_lock = threading.Lock()
_sync_pool: ConnectionPool | None = None
_async_pool: AsyncConnectionPool | None = None
_async_lock: asyncio.Lock | None = None


def _conn_kwargs() -> dict:
    return {
        "autocommit": True,
        "row_factory": _row_factory,
        # Fixed zone so timestamps serialise identically everywhere; a statement
        # that runs away is cut off instead of holding a pooled connection forever.
        "options": "-c timezone=UTC -c statement_timeout=30000",
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 3,
    }


def _url() -> str:
    return os.environ["DATABASE_URL"]


def pool() -> ConnectionPool:
    global _sync_pool
    if _sync_pool is None:
        with _lock:
            if _sync_pool is None:
                p = ConnectionPool(
                    _url(), min_size=1, max_size=_SYNC_MAX, kwargs=_conn_kwargs(),
                    open=False, timeout=10, max_idle=300, max_lifetime=3600,
                    check=ConnectionPool.check_connection, name="omni-sync",
                )
                p.open(wait=True, timeout=15)
                _sync_pool = p
    return _sync_pool


async def apool() -> AsyncConnectionPool:
    global _async_pool, _async_lock
    if _async_pool is None:
        if _async_lock is None:
            _async_lock = asyncio.Lock()
        async with _async_lock:
            if _async_pool is None:
                p = AsyncConnectionPool(
                    _url(), min_size=1, max_size=_ASYNC_MAX, kwargs=_conn_kwargs(),
                    open=False, timeout=10, max_idle=300, max_lifetime=3600,
                    check=AsyncConnectionPool.check_connection, name="omni-async",
                )
                await p.open(wait=True, timeout=15)
                _async_pool = p
    return _async_pool


def close_pools() -> None:
    """Sync part of shutdown; the async pool is closed by ``aclose_pools``."""
    global _sync_pool
    if _sync_pool is not None:
        _sync_pool.close()
        _sync_pool = None


async def aclose_pools() -> None:
    global _async_pool
    if _async_pool is not None:
        await _async_pool.close()
        _async_pool = None
    close_pools()


# ── query helpers (sync) ────────────────────────────────────────────────────


def fetch_all(query: str | sql.Composable, params: Any = None) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(query, params).fetchall()


def fetch_one(query: str | sql.Composable, params: Any = None) -> dict | None:
    with pool().connection() as conn:
        return conn.execute(query, params).fetchone()


def execute(query: str | sql.Composable, params: Any = None) -> int:
    """Run a statement; returns the number of rows affected."""
    with pool().connection() as conn:
        return conn.execute(query, params).rowcount


# ── query helpers (async) ───────────────────────────────────────────────────


async def afetch_all(query: str | sql.Composable, params: Any = None) -> list[dict]:
    async with (await apool()).connection() as conn:
        cur = await conn.execute(query, params)
        return await cur.fetchall()


async def afetch_one(query: str | sql.Composable, params: Any = None) -> dict | None:
    async with (await apool()).connection() as conn:
        cur = await conn.execute(query, params)
        return await cur.fetchone()


async def aexecute(query: str | sql.Composable, params: Any = None) -> int:
    async with (await apool()).connection() as conn:
        cur = await conn.execute(query, params)
        return cur.rowcount


# ── dict -> SQL builders ────────────────────────────────────────────────────
# For the eval tooling and other places that move whole rows around. Identifiers
# are quoted by psycopg; values are always bound parameters.


def _where(where: dict[str, Any]) -> tuple[sql.Composable, list]:
    parts, params = [], []
    for col, val in where.items():
        if isinstance(val, (list, tuple, set)):
            parts.append(sql.SQL("{} = ANY(%s)").format(sql.Identifier(col)))
            params.append(list(val))
        elif val is None:
            parts.append(sql.SQL("{} IS NULL").format(sql.Identifier(col)))
        else:
            parts.append(sql.SQL("{} = %s").format(sql.Identifier(col)))
            params.append(val)
    return sql.SQL(" AND ").join(parts) if parts else sql.SQL("TRUE"), params


def _insert_sql(table: str, rows: list[dict]) -> tuple[sql.Composed, list, list[str]]:
    """Multi-row INSERT. Rows may omit columns that other rows have: a missing
    column gets the column's DEFAULT."""
    cols: list[str] = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    row_phs = [
        sql.SQL("({})").format(
            sql.SQL(", ").join(sql.Placeholder() if c in r else sql.SQL("DEFAULT") for c in cols)
        )
        for r in rows
    ]
    stmt = sql.SQL("INSERT INTO {} ({}) VALUES {}").format(
        sql.Identifier(table),
        sql.SQL(", ").join(sql.Identifier(c) for c in cols),
        sql.SQL(", ").join(row_phs),
    )
    params = [adapt(r[c]) for r in rows for c in cols if c in r]
    return stmt, params, cols


def insert(table: str, rows: dict | list[dict]) -> list[dict]:
    """INSERT one row or many; returns the inserted rows (RETURNING *)."""
    rows = [rows] if isinstance(rows, dict) else list(rows)
    if not rows:
        return []
    stmt, params, _ = _insert_sql(table, rows)
    return fetch_all(stmt + sql.SQL(" RETURNING *"), params)


def upsert(
    table: str, rows: dict | list[dict], conflict: str | Iterable[str], update: Iterable[str] | None = None
) -> list[dict]:
    """INSERT ... ON CONFLICT (conflict) DO UPDATE. By default every non-key column
    is overwritten with the incoming value."""
    rows = [rows] if isinstance(rows, dict) else list(rows)
    if not rows:
        return []
    key = [conflict] if isinstance(conflict, str) else list(conflict)
    stmt, params, cols = _insert_sql(table, rows)
    upd = list(update) if update is not None else [c for c in cols if c not in key]
    tail = sql.SQL(" ON CONFLICT ({}) ").format(sql.SQL(", ").join(sql.Identifier(c) for c in key))
    if upd:
        tail += sql.SQL("DO UPDATE SET {}").format(
            sql.SQL(", ").join(sql.SQL("{0} = EXCLUDED.{0}").format(sql.Identifier(c)) for c in upd)
        )
    else:
        tail += sql.SQL("DO NOTHING")
    return fetch_all(stmt + tail + sql.SQL(" RETURNING *"), params)


def update(table: str, patch: dict[str, Any], where: dict[str, Any]) -> list[dict]:
    """UPDATE table SET patch WHERE where (equality / IN); returns the updated rows."""
    if not patch:
        return []
    sets = sql.SQL(", ").join(sql.SQL("{} = %s").format(sql.Identifier(c)) for c in patch)
    cond, wparams = _where(where)
    stmt = sql.SQL("UPDATE {} SET {} WHERE {} RETURNING *").format(sql.Identifier(table), sets, cond)
    return fetch_all(stmt, [adapt(v) for v in patch.values()] + wparams)


def delete(table: str, where: dict[str, Any]) -> list[dict]:
    """DELETE FROM table WHERE where; returns the deleted rows."""
    cond, params = _where(where)
    return fetch_all(sql.SQL("DELETE FROM {} WHERE {} RETURNING *").format(sql.Identifier(table), cond), params)


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)
