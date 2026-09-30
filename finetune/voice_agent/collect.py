"""Collect gpt-6-luna's answers to `queries.yaml` as the voice agent would give them.

    python finetune/voice_agent/collect.py            # all queries
    python finetune/voice_agent/collect.py --limit 4  # quick look
    python finetune/voice_agent/collect.py --ids zh-stock-03,en-omni-05  # re-sample just these

Each query runs through the *production* live-call voice agent — same system
prompt (`VOICE_SYSTEM_PROMPT + VOICE_CALL_PROMPT_ADDENDUM`), same tools, same
`ToolCallLimitMiddleware(run_limit=2)`, same turn framing
(`core.voice.agent._build_turn_content`) — with only the model swapped for
`gpt_6_luna_voice` and the checkpointer dropped (every query is a fresh
one-turn thread). Tools are real: web search, weather and python hit their
live backends, so tool results in the traces are what a call would really see.

Writes `dataset/traces.jsonl`, one record per query:

    {id, lang, cat, text, messages: [system, user, assistant, tool, ...],
     tools: [OpenAI schemas], tool_calls: [...], tool_errors: int, error: str|None,
     source: "hand" only when queries.yaml gives an `answer:`}

`messages` is OpenAI chat format (assistant `tool_calls[].function.arguments`
is a JSON string), the shape `art.utils.sft.train_sft_from_file` reads.
Filtering happens in `filter.py`, not here — this file keeps everything.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import dotenv
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
dotenv.load_dotenv(ROOT / ".env")

from langchain.agents import create_agent  # noqa: E402
from langchain.agents.middleware import ToolCallLimitMiddleware  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.utils.function_calling import convert_to_openai_tool  # noqa: E402

from core.llm import gpt_6_luna_voice  # noqa: E402
from core.stream import _text_of  # noqa: E402
from core.voice.agent import VOICE_TOOLS, _build_turn_content, end_call  # noqa: E402
from core.voice.prompt import VOICE_CALL_PROMPT_ADDENDUM, VOICE_SYSTEM_PROMPT  # noqa: E402

sys.path.insert(0, str(HERE))
from filter import annotate, read_jsonl  # noqa: E402

DATA = HERE / "dataset"
# The flags a --ids re-sample tries to clear. Length is left out: a re-sample
# is for characters and identity the model shouldn't have produced, not for
# re-rolling until the answer happens to be short.
RESAMPLE_ON = {"markup", "identity", "tool_error", "empty"}
SYSTEM_PROMPT = VOICE_SYSTEM_PROMPT + VOICE_CALL_PROMPT_ADDENDUM

# What a client would report. Chinese queries lean towards Chinese cities.
_CITIES_ZH = ["Shanghai, China", "Beijing, China", "Guangzhou, China", "Chengdu, China", "Shenzhen, China"]
_CITIES_EN = [
    "Chicago, Illinois, United States", "San Francisco, California, United States",
    "New York, New York, United States", "Champaign, Illinois, United States",
    "London, United Kingdom", "Seattle, Washington, United States",
]


def turn_context(q: dict, rng: random.Random) -> tuple[str | None, str]:
    """(user_location, user_local_datetime) for one query. `loc: none` sends no
    location — "what's the weather" with no city is a case the model must handle."""
    if q.get("loc") == "none":
        location = None
    else:
        own, other = (_CITIES_ZH, _CITIES_EN) if q["lang"] == "zh" else (_CITIES_EN, _CITIES_ZH)
        location = rng.choice(own if rng.random() < 0.8 else other)
    return location, datetime.now().isoformat(timespec="seconds")


def _agent():
    return create_agent(
        model=gpt_6_luna_voice,
        tools=[*VOICE_TOOLS, end_call],
        system_prompt=SYSTEM_PROMPT,
        middleware=[ToolCallLimitMiddleware(run_limit=2)],
    )


def _to_openai(messages: list) -> tuple[list[dict], list[dict], int]:
    """LangChain messages after the user turn -> OpenAI chat dicts, plus a flat
    tool-call list and the count of tool results that look like failures."""
    out, calls, errors = [], [], 0
    for m in messages:
        if isinstance(m, AIMessage):
            msg: dict = {"role": "assistant", "content": _text_of(m.content).strip()}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c["args"] or {}, ensure_ascii=False)},
                    }
                    for c in m.tool_calls
                ]
                calls += [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
            out.append(msg)
        elif isinstance(m, ToolMessage):
            body = _text_of(m.content)
            if m.status == "error" or "limit exceeded" in body.lower() or "unavailable" in body.lower()[:80]:
                errors += 1
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": body})
    return out, calls, errors


async def collect_one(agent, tools: list[dict], q: dict, rng: random.Random) -> dict:
    location, dt = turn_context(q, rng)
    user = _build_turn_content(q["text"], location, dt)
    rec = {
        "id": q["id"], "lang": q["lang"], "cat": q["cat"], "text": q["text"],
        "user_location": location, "user_local_datetime": dt,
        "tools": tools, "error": None,
    }
    if q.get("answer"):
        # Hand-written: same system prompt and turn framing as any other row, no model.
        rec.update(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user},
                {"role": "assistant", "content": q["answer"]},
            ],
            tool_calls=[], tool_errors=0, source="hand",
        )
        return rec
    try:
        result = await agent.ainvoke({"messages": [HumanMessage(user)]})
    except Exception as e:  # noqa: BLE001 — a failed query is a record, not a crash
        rec.update(messages=[], tool_calls=[], tool_errors=0, error=f"{type(e).__name__}: {e}")
        return rec
    msgs = result["messages"]
    body, calls, errors = _to_openai(msgs[1:])  # msgs[0] is the user turn
    rec["messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}, *body,
    ]
    rec.update(tool_calls=calls, tool_errors=errors)
    return rec


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", default="",
                    help="comma-separated query ids to re-sample; the fresh records replace "
                         "the same ids in traces.jsonl and the rest are kept")
    ap.add_argument("--attempts", type=int, default=3,
                    help="with --ids: sample each up to N times, stopping at the first record "
                         "with no markup / identity / tool_error flag")
    args = ap.parse_args()

    queries = yaml.safe_load((HERE / "queries.yaml").read_text(encoding="utf-8"))
    resample = {i for i in args.ids.split(",") if i}
    if resample and args.ids == "hand":
        resample = {q["id"] for q in queries if q.get("answer")}
        args.ids = ",".join(sorted(resample))
    if resample:
        queries = [q for q in queries if q["id"] in resample]
        if len(queries) != len(resample):
            raise SystemExit(f"unknown ids: {resample - {q['id'] for q in queries}}")
    elif args.limit:
        # Spread the limit across the list rather than taking the first N (all zh chat).
        queries = queries[:: max(1, len(queries) // args.limit)][: args.limit]

    agent = _agent()
    tools = [convert_to_openai_tool(t) for t in [*VOICE_TOOLS, end_call]]
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()

    async def run(q: dict) -> dict:
        ctx_rng = random.Random(f"{args.seed}-{q['id']}")  # per-query, so order/concurrency can't change it
        async with sem:
            for attempt in range(1, (args.attempts if resample else 1) + 1):
                rec = await collect_one(agent, tools, q, ctx_rng)
                if not resample:
                    break
                bad = [f for f in annotate(rec)["flags"] if f.split(":")[0] in RESAMPLE_ON]
                print(f"  {q['id']:<16} attempt {attempt}: {' '.join(bad) or 'clean'}", flush=True)
                if not bad:
                    break
        status = rec["error"] or f"{len(rec['tool_calls'])} tool call(s)"
        print(f"  {q['id']:<16} {status}", flush=True)
        return rec

    records = await asyncio.gather(*(run(q) for q in queries))
    DATA.mkdir(exist_ok=True)
    out = DATA / "traces.jsonl"
    if resample:
        fresh = {r["id"]: r for r in records}
        old = read_jsonl(out)
        # Replace in place, and append ids the file has never seen (new queries).
        records = [fresh.get(r["id"], r) for r in old] + [r for i, r in fresh.items() if i not in {o["id"] for o in old}]
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False).replace("\u2028", "\\u2028").replace("\u2029", "\\u2029") + "\n")
    failed = sum(1 for r in records if r["error"])
    print(f"\n{len(records)} records ({failed} failed) -> {out} in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
