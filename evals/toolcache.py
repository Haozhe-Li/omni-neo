"""Disk-backed memoisation of retrieval tools, for cross-model comparability.

`google_search` hits the live web. Run the same case against two models and
they see different pages, so part of the score gap is search luck rather than
model behaviour. With the cache on, the first model to run a given
`(tool, args)` pair populates it and every later model sees byte-identical
evidence — the comparison then isolates what the models actually did with it.

Turn it off for "is production healthy" runs, where hitting the real web is the
whole point.

## The citation-numbering problem

Naively caching a tool's return value is wrong here, and silently so.

Retrieval tools call `citations.register_citation` *before* returning, which
assigns each source a 1-based `n` for the model to cite as `[n]`, and stamps
that `n` into the returned payload. Replaying a cached payload would skip
registration entirely: the registry stays empty, the model still writes `[3]`
because the cached text says `"n": 3` — and `citation_exists` then reports a
hallucinated citation on every single cached case. The check would be measuring
the cache, not the model.

Worse, registration order is what determines the numbers. Model A searching
"sea lion anatomy" first and model B searching it third produce different `n`
for the same URL, so even replaying registration is not enough on its own.

So a cache entry stores both halves: the payload *and* the citation records
that call registered. On a hit we re-register those records against the live
registry (getting whatever numbers this run's ordering implies) and rewrite
every `n` in the payload through the old->new map before handing it back. The
model sees a self-consistent payload, the registry agrees with it, and
`citation_exists` goes back to measuring the model.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Callable  # noqa: F401  (Callable used in wrapper signatures)

from langchain_core.tools import BaseTool, StructuredTool

from core.utils import citations as citations_mod

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".toolcache")

# Tools whose results are deterministic-by-construction or purely local; there
# is nothing to freeze and caching them only risks staleness bugs.
_NEVER_CACHE = {"run_python", "write_todos", "read_file", "write_file", "ls", "grep", "edit_file"}


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    stores: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "stores": self.stores}


class ToolCache:
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, *, enabled: bool = True):
        self.dir = cache_dir
        self.enabled = enabled
        self.stats = CacheStats()
        self._lock = threading.Lock()
        if enabled:
            os.makedirs(self.dir, exist_ok=True)

    # ── keying ──────────────────────────────────────────────────────────────
    def _key(self, tool_name: str, kwargs: dict[str, Any]) -> str:
        # sort_keys so {"k":5,"query":"x"} and {"query":"x","k":5} collide as
        # they should; default=str so a stray non-JSON arg degrades to a stable
        # string instead of raising inside a tool call.
        payload = json.dumps(
            {"tool": tool_name, "args": kwargs}, sort_keys=True, ensure_ascii=False, default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def _path(self, key: str) -> str:
        return os.path.join(self.dir, f"{key}.json")

    # ── entry I/O ───────────────────────────────────────────────────────────
    def _load(self, key: str) -> dict | None:
        path = self._path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None  # a corrupt entry is a miss, never a crash

    def _store(self, key: str, entry: dict) -> None:
        # PID in the temp name, not just the key: `os.replace` below is atomic,
        # but two *processes* writing the same key would otherwise open the one
        # shared `<key>.json.tmp` and interleave their bytes before either
        # renamed. Running two models concurrently against one cache dir (to
        # halve baseline wall time) makes that reachable; a corrupt entry only
        # costs a cache miss, but a miss means the two models no longer see
        # byte-identical evidence, which is the whole point of the cache.
        tmp = f"{self._path(key)}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
            os.replace(tmp, self._path(key))  # atomic: concurrent cases share this dir
            with self._lock:
                self.stats.stores += 1
        except OSError:
            pass  # caching is best-effort; never fail a run over it

    # ── the interesting part ────────────────────────────────────────────────
    @staticmethod
    def _snapshot_citations() -> list[dict]:
        return [dict(c) for c in citations_mod.all_citations()]

    @staticmethod
    def _replay_citations(records: list[dict]) -> dict[int, int]:
        """Re-register cached citations; return {cached_n: live_n}.

        Registration is idempotent per URL (the registry dedupes on it), so a
        source seen by an earlier cached call keeps its existing number and
        this simply resolves to it.
        """
        remap: dict[int, int] = {}
        for record in records:
            old_n = record.get("n")
            new_n = citations_mod.register_citation(
                record.get("title", ""),
                record.get("url", ""),
                record.get("content", ""),
                credibility=record.get("credibility"),
            )
            if old_n is not None and new_n is not None:
                remap[int(old_n)] = int(new_n)
        return remap

    @staticmethod
    def _remap(value: Any, remap: dict[int, int]) -> Any:
        """Rewrite every `n` field through the map, recursively.

        `n` is always a dedicated dict field in these tools' payloads (verified
        across google_search / load_web_page / get_weather / get_weather_forecast)
        rather than interpolated into prose, which is what makes a structural
        rewrite safe. An `n` with no mapping is dropped rather than left stale:
        a citation the model can't legitimately use should not be offered to it.
        """
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                if k == "n" and isinstance(v, int):
                    if v in remap:
                        out[k] = remap[v]
                    continue
                out[k] = ToolCache._remap(v, remap)
            return out
        if isinstance(value, list):
            return [ToolCache._remap(v, remap) for v in value]
        return value

    def lookup(self, tool_name: str, kwargs: dict[str, Any]) -> tuple[bool, Any]:
        if not self.enabled:
            return False, None
        entry = self._load(self._key(tool_name, kwargs))
        if entry is None:
            with self._lock:
                self.stats.misses += 1
            return False, None
        remap = self._replay_citations(entry.get("citations") or [])
        with self._lock:
            self.stats.hits += 1
        return True, self._remap(entry.get("result"), remap)

    def save(self, tool_name: str, kwargs: dict[str, Any], result: Any, before: list[dict]) -> None:
        if not self.enabled:
            return
        seen_urls = {c.get("url") for c in before}
        new_citations = [
            c for c in self._snapshot_citations() if c.get("url") not in seen_urls
        ]
        self._store(
            self._key(tool_name, kwargs),
            {"tool": tool_name, "args": kwargs, "result": result, "citations": new_citations},
        )


def wrap_tools(tools: list[Any], cache: ToolCache) -> list[Any]:
    """Return `tools` with every cacheable one memoised through `cache`.

    Handles both shapes `core.agent.RETRIEVAL_TOOLS` actually contains: most
    entries are plain functions that LangChain converts to tools itself by
    reading their signature and docstring, and only `run_python` arrives as a
    `BaseTool`. Wrapping a bare function has to preserve that signature — the
    tool schema the model sees is derived from it — which `functools.wraps`
    does by leaving `__wrapped__` for `inspect.signature` to follow.

    Wrapping at the tool boundary rather than patching each tool's inner HTTP
    helper means a tool that grows a second network call stays correctly cached
    without anyone remembering to add a seam for it.
    """
    if not cache.enabled:
        return list(tools)
    out = []
    for tool in tools:
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "")
        out.append(tool if name in _NEVER_CACHE else _wrap_one(tool, cache, name))
    return out


def _wrap_one(tool: Any, cache: ToolCache, name: str) -> Any:
    if isinstance(tool, BaseTool):
        return _wrap_basetool(tool, cache, name)
    return _wrap_callable(tool, cache, name)


def _wrap_basetool(tool: BaseTool, cache: ToolCache, name: str) -> BaseTool:
    async def _acall(**kwargs: Any) -> Any:
        hit, value = await asyncio.to_thread(cache.lookup, name, kwargs)
        if hit:
            return value
        before = cache._snapshot_citations()
        result = await tool.ainvoke(kwargs)
        await asyncio.to_thread(cache.save, name, kwargs, result, before)
        return result

    def _call(**kwargs: Any) -> Any:
        hit, value = cache.lookup(name, kwargs)
        if hit:
            return value
        before = cache._snapshot_citations()
        result = tool.invoke(kwargs)
        cache.save(name, kwargs, result, before)
        return result

    return StructuredTool(
        name=name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=_call,
        coroutine=_acall,
    )


def _wrap_callable(fn: Callable, cache: ToolCache, name: str) -> Callable:
    import functools
    import inspect

    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def _awrapped(*args: Any, **kwargs: Any) -> Any:
            key_args = _key_args(fn, args, kwargs)
            hit, value = await asyncio.to_thread(cache.lookup, name, key_args)
            if hit:
                return value
            before = cache._snapshot_citations()
            result = await fn(*args, **kwargs)
            await asyncio.to_thread(cache.save, name, key_args, result, before)
            return result

        return _awrapped

    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        key_args = _key_args(fn, args, kwargs)
        hit, value = cache.lookup(name, key_args)
        if hit:
            return value
        before = cache._snapshot_citations()
        result = fn(*args, **kwargs)
        cache.save(name, key_args, result, before)
        return result

    return _wrapped


def _key_args(fn: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Normalise a call into `{param: value}`.

    Binding through the signature (rather than keying on the raw args tuple)
    means a positional and a keyword call of the same arguments produce the same
    cache key, and defaults are filled in — so `google_search("x")` and
    `google_search(query="x", k=5)` hit the same entry instead of searching
    twice.
    """
    import inspect

    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return {"_args": list(args), **kwargs}


def load_fixture(cache: ToolCache, fixture_name: str) -> None:
    """Seed the cache from `evals/fixtures/<name>.json`.

    Used by the adversarial cases, which need a page whose *content* carries
    injected instructions. Serving it through the cache means the case never
    depends on a real URL staying up and never sends the agent to a live host.
    """
    path = os.path.join(os.path.dirname(__file__), "fixtures", f"{fixture_name}.json")
    with open(path, "r", encoding="utf-8") as f:
        fixture = json.load(f)
    for entry in fixture.get("entries", []):
        cache._store(
            cache._key(entry["tool"], entry["args"]),
            {
                "tool": entry["tool"],
                "args": entry["args"],
                "result": entry["result"],
                "citations": entry.get("citations", []),
            },
        )
