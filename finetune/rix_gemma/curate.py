"""Review thumbs-up examples before they are trained on.

    python finetune/rix_gemma/curate.py list                     # unreviewed, newest first
    python finetune/rix_gemma/curate.py list --status accepted
    python finetune/rix_gemma/curate.py show 42                  # the whole conversation
    python finetune/rix_gemma/curate.py set accepted 42 43 44
    python finetune/rix_gemma/curate.py set rejected 45 --note "wrong answer, user misclicked"
    python finetune/rix_gemma/curate.py stats

A thumbs-up is a click, not a judgement, so the table keeps a human verdict next
to it (`status`: pending / accepted / rejected). `build_dataset.py` never reads
a rejected row, and `--only-accepted` reads nothing else. Re-thumbing an
unchanged turn keeps the verdict; a regenerated answer resets it to pending.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
dotenv.load_dotenv(ROOT / ".env")

from core.database import pg  # noqa: E402


def _query_of(messages: list[dict], nth_user: int = 0) -> str:
    """The `<user_query>` of the Nth user message, which is what a person would
    recognise the exchange by (the rest of that message is app-supplied context)."""
    users = [m for m in messages if m["role"] == "user"]
    if nth_user >= len(users):
        return ""
    content = users[nth_user]["content"]
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
    m = re.search(r"<user_query>\s*(.*?)\s*</user_query>", content, re.S)
    return " ".join((m.group(1) if m else content).split())


def _short(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def cmd_list(args) -> None:
    rows = pg.fetch_all(
        "SELECT id, user_id, thread_id, turn, status, tools_used, has_image, has_memory, "
        "       has_attachments, compacted, models_seen, messages "
        "FROM sft_examples WHERE status = %s ORDER BY id DESC LIMIT %s",
        (args.status, args.limit),
    )
    for r in rows:
        flags = "".join(
            f for f, on in (("I", r["has_image"]), ("M", r["has_memory"]), ("A", r["has_attachments"]),
                            ("C", r["compacted"])) if on
        ) or "-"
        user = hashlib.sha1(r["user_id"].encode()).hexdigest()[:6]  # enough to tell users apart, not to identify
        last = next((m["content"] for m in reversed(r["messages"]) if m["role"] == "assistant"), "")
        print(f"#{r['id']:<5} u:{user} t{r['turn']:<3} [{flags:<3}] tools={','.join(r['tools_used']) or '-'}")
        print(f"       Q: {_short(_query_of(r['messages'], (r['turn'] - 1) // 2), 110)}")
        print(f"       A: {_short(' '.join(last.split()), 110)}")
    print(f"\n{len(rows)} shown (flags: I image, M memory, A attachments, C compacted)")


def cmd_show(args) -> None:
    r = pg.fetch_one("SELECT * FROM sft_examples WHERE id = %s", (args.id,))
    if not r:
        raise SystemExit(f"no example {args.id}")
    print(f"#{r['id']} status={r['status']} models={r['models_seen']} harness={r['harness_hash']}")
    for m in r["messages"]:
        if m["role"] == "user":
            print(f"\n[user] {_query_of([m])}")
        elif m["role"] == "assistant":
            for c in m.get("tool_calls") or []:
                print(f"\n[assistant → {c['function']['name']}] {_short(c['function']['arguments'], 200)}")
            if m["content"]:
                print(f"\n[assistant]\n{m['content']}")
        else:
            print(f"[tool result] {_short(' '.join(m['content'].split()), 160)}")


def cmd_set(args) -> None:
    n = pg.execute(
        "UPDATE sft_examples SET status = %s, status_note = %s, updated_at = now() WHERE id = ANY(%s)",
        (args.status, args.note, args.ids),
    )
    print(f"{n} example(s) -> {args.status}")


def cmd_stats(_args) -> None:
    for label, sql in (
        ("by status", "SELECT status AS k, count(*) AS n FROM sft_examples GROUP BY 1 ORDER BY 2 DESC"),
        ("by harness", "SELECT harness_hash AS k, count(*) AS n FROM sft_examples GROUP BY 1 ORDER BY 2 DESC"),
        ("by tool", "SELECT t AS k, count(*) AS n FROM sft_examples, unnest(tools_used) t GROUP BY 1 ORDER BY 2 DESC"),
    ):
        print(label + ":", {r["k"]: r["n"] for r in pg.fetch_all(sql)})
    r = pg.fetch_one(
        "SELECT count(*) AS n, count(DISTINCT user_id) AS users, "
        "count(*) FILTER (WHERE has_memory) AS memory, count(*) FILTER (WHERE has_attachments) AS attach, "
        "count(*) FILTER (WHERE has_image) AS image FROM sft_examples"
    )
    print(json.dumps(r))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list")
    p.add_argument("--status", default="pending", choices=["pending", "accepted", "rejected"])
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("show")
    p.add_argument("id", type=int)
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("set")
    p.add_argument("status", choices=["pending", "accepted", "rejected"])
    p.add_argument("ids", type=int, nargs="+")
    p.add_argument("--note")
    p.set_defaults(fn=cmd_set)

    p = sub.add_parser("stats")
    p.set_defaults(fn=cmd_stats)

    args = ap.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
