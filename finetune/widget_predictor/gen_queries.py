"""Generate candidate queries for the widget-predictor training set.

Writing a few hundred queries by hand produces a set that all sounds like one
person on one afternoon — same phrasings, same cities, same tickers. Generating
them per category with an explicit avoid-list gives a wider spread, and every
candidate still has to survive dedup, the production word-count gate, and the
teacher-labelling pass before it becomes training data.

This produces *unlabelled* queries only. Labels come from
`label.py`; nothing here decides what the right answer is.

    python finetune/widget_predictor/gen_queries.py -n 500
"""
from __future__ import annotations

import argparse
import asyncio
import re
import sys
import unicodedata
from pathlib import Path

import dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DATA = Path(__file__).resolve().parent / "dataset"
dotenv.load_dotenv()

from core import llm  # noqa: E402
from core.widget_predictor import (  # noqa: E402
    _PREDICTOR_PROMPT,
    _WORD_LIMIT,
    _word_count,
    response_text,
)

# Category -> (share of the total, what to ask for). The shares mirror the
# reviewed 100-row ground truth, with negatives kept dominant because a
# spurious widget is the failure the user actually sees.
CATEGORIES: dict[str, tuple[float, str]] = {
    "weather": (0.16, """\
Queries asking for current weather or a forecast. Vary the phrasing wildly: \
bare city names with a weather word, "will it rain", "帮我看下天气", temperature \
questions, weekend forecasts, "外面冷吗". Use cities from every continent, not \
just the famous ten. About a third should rely on an implicit location \
("今天要穿外套吗", "is it windy out") — for those, append a TAB and a plausible \
user location like "Berlin, Germany"."""),
    "stock": (0.15, """\
Queries about a specific public company's share price, earnings or market \
performance. Mix bare tickers ("amzn"), company names in English and Chinese, \
US and non-US listings (Hong Kong, Tokyo, Taiwan, Seoul, Frankfurt, London), \
ETFs and indices, and casual phrasings like "how is X doing", "X 跌了吗", \
"did nvidia go up"."""),
    "currency": (0.07, """\
Queries converting or comparing two national currencies. Vary between explicit \
amounts ("250 cad in eur"), bare pairs ("gbp jpy"), and Chinese phrasings \
("换日元合适吗"). Include a few that rely on the user's location to infer one \
side ("欧元汇率") — append a TAB and a user location for those. Only real fiat \
currencies, never crypto."""),
    "entity": (0.28, """\
Queries whose topic is ONE named entity — a person, company, product, country, \
or landmark. Include both definitional ("what is X", "X是什么") and attribute \
questions ("who founded X", "X 多高", "is X profitable", "when was X built"). \
Range across tech companies, historical figures, musicians, athletes, consumer \
products, countries, cities and landmarks. Non-Western entities too."""),
    "hard_negative": (0.22, """\
Queries that LOOK like they need a live-data card but must not get one. Mix: \
comparisons of two named things ("X vs Y", "X和Y哪个好"); how-to and procedural \
questions that mention a product ("how to cancel netflix", "微信怎么退群"); \
questions about abstract concepts or categories ("what is inflation", "best \
budget phones"); questions mentioning a currency or stock generically ("should \
I invest in index funds", "如何看懂财报"); questions mentioning a place for a \
non-weather reason ("cheap flights to lisbon", "东京地铁怎么坐"); and crypto \
prices, which we have no widget for."""),
    "easy_negative": (0.12, """\
Everyday assistant requests with no live-data angle at all: writing help, \
coding help, translation, summarising, math, advice, small talk, personal \
feelings. Half in Chinese, half in English."""),
}

_PROMPT = """\
You are helping build a test set for a search assistant's widget router.

Write {n} DISTINCT user queries in this category:
{spec}

Rules:
- Write the way people actually type into a search box: short, lowercase, often \
sentence fragments, no trailing punctuation. A few may have typos.
- {lang}
- Chinese queries must be at most 10 Chinese characters. This is a hard limit.
- English queries must be at most 10 words.
- Output ONE query per line, nothing else — no numbering, no bullets, no \
explanation, no quotes around them.
- If a query needs a user location to make sense, put a TAB then the location \
after it. Otherwise no tab.

Do NOT repeat or lightly reword any of these, which already exist:
{avoid}
"""

_PUNCT_RE = re.compile(r"[\s　!-/:-@\[-`{-~，。！？、；：""''（）【】]+")


def normalize(query: str) -> str:
    """Aggressive key for dedup: casing, spacing and punctuation all collapse."""
    folded = unicodedata.normalize("NFKC", query).casefold()
    return _PUNCT_RE.sub("", folded)


async def generate(category: str, spec: str, want: int, model, batch: int) -> list[str]:
    """Generate `want` queries for one category, growing the avoid-list as we go."""
    seen: dict[str, str] = {}
    rows: list[str] = []
    attempts = 0
    while len(rows) < want and attempts < 12:
        attempts += 1
        lang = ("Write them all in Chinese." if attempts % 3 == 2 else
                "Write them all in English." if attempts % 3 == 1 else
                "Mix Chinese and English roughly half and half.")
        avoid = "\n".join(list(seen.values())[-120:]) or "(nothing yet)"
        resp = await model.ainvoke([
            ("user", _PROMPT.format(n=batch, spec=spec, lang=lang, avoid=avoid))
        ])
        for line in response_text(resp).splitlines():
            line = line.strip().lstrip("-*0123456789. ").strip()
            if not line:
                continue
            query, _, location = line.partition("\t")
            query = query.strip().strip('"\'')
            if not query or _word_count(query) > _WORD_LIMIT:
                continue
            key = normalize(query)
            if not key or key in seen:
                continue
            seen[key] = query
            rows.append(f"{query}\t{location.strip()}" if location.strip() else query)
            if len(rows) >= want:
                break
        print(f"  {category:<14} {len(rows)}/{want}")
    return rows


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--total", type=int, default=500)
    ap.add_argument("-o", "--output", type=Path, default=DATA / "queries_new.tsv")
    # Excluding against the labelled set, not just the hand-written seeds:
    # every query already labelled is off limits, whichever round produced it.
    ap.add_argument("--exclude", type=Path, nargs="*",
                    default=[DATA / "ground_truth.tsv"],
                    help="existing query files whose entries must not recur")
    ap.add_argument("--model", default="gemini_flash")
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()

    model = getattr(llm, args.model)

    # Anything already in an existing set is off limits: a query that appears in
    # both the train and test split turns the eval into a memorisation check.
    # The system prompt's own few-shot examples are excluded for the same
    # reason — they ship with the answer attached, so scoring a model on one
    # measures whether it can copy from its own context.
    existing = {normalize(q) for q in re.findall(r"^Query: (.+)$", _PREDICTOR_PROMPT, re.M)}
    for path in args.exclude:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                existing.add(normalize(line.split("\t")[0]))
    print(f"excluding {len(existing)} existing queries\n")

    results = await asyncio.gather(*(
        generate(name, spec, round(share * args.total), model, args.batch)
        for name, (share, spec) in CATEGORIES.items()
    ))

    lines: list[str] = [
        "# Generated candidate queries for the widget predictor (unlabelled).",
        f"# {args.model}, deduped against {', '.join(str(p) for p in args.exclude)}.",
        "# Grouped by the category they were generated for — that is a generation",
        "# hint only, NOT a label. Labels come from scripts/label_widgets.py.",
    ]
    kept: set[str] = set()
    total = 0
    for (name, _), rows in zip(CATEGORIES.items(), results):
        lines.append(f"\n# ── {name} ──")
        for row in rows:
            key = normalize(row.split("\t")[0])
            if key in existing or key in kept:
                continue
            kept.add(key)
            lines.append(row)
            total += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {total} queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
