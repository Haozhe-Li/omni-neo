"""Pure text extraction — no scoring, no I/O.

Everything the deterministic checks need to know about an answer is derived
here: the `<report>` block, the ```echarts / ```map fences, the `<question>`
block, citation markers, word counts. Kept separate from `checks.py` so the
parsing can be unit-tested against real model output without dragging in the
check registry.

The block syntaxes mirror what the frontend actually parses (see the charting /
mapping / report-writing / ask-question skills). Where the frontend is
forgiving, this is too; where it is strict, so is this — the point is to fail a
check exactly when the real UI would fail to render.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ── word counting ───────────────────────────────────────────────────────────
# CJK text has no spaces, so `len(text.split())` reports a 900-character
# Chinese report as ~3 "words" and every min_words check passes or fails at
# random. Count CJK codepoints individually and Latin runs as words, then add.
_CJK_RE = re.compile(
    r"[㐀-䶿一-鿿豈-﫿぀-ヿ가-힯]"
)
_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")


def count_words(text: str) -> int:
    if not text:
        return 0
    return len(_CJK_RE.findall(text)) + len(_LATIN_WORD_RE.findall(text))


# ── language detection ──────────────────────────────────────────────────────
# Deliberately a character-class ratio rather than a language-ID model: the
# only distinction that matters here is Chinese vs English, the signal is
# unambiguous at the codepoint level, and a deterministic check must not depend
# on a model download or a network call.
#
# The thresholds are wide apart on purpose, because both languages legitimately
# borrow from the other. An English answer explaining what 你好 means still
# quotes CJK; a Chinese answer about LangChain valuations is full of Latin
# product names and numbers. Anything landing between the two thresholds is
# reported as "mixed" rather than forced into a bucket — a genuinely
# half-and-half answer is a real failure worth surfacing, not a coin flip.
# Because CJK is counted per character while English is counted per word, real
# Chinese prose sits far above this floor even when dense with English product
# names and numbers (measured: ~0.7-0.9). The floor is set well clear of 0.5 so
# that a genuinely half-and-half answer lands in the "mixed" band instead of
# being rounded into whichever language won by a character or two.
_ZH_FLOOR = 0.55   # at or above -> Chinese
_EN_CEIL = 0.15    # at or below -> English


@dataclass
class LanguageVerdict:
    lang: str        # zh | en | mixed | unknown
    cjk_chars: int
    latin_words: int
    ratio: float

    def __str__(self) -> str:
        return f"{self.lang} (cjk={self.cjk_chars}, latin={self.latin_words}, ratio={self.ratio:.2f})"


def detect_language(text: str) -> LanguageVerdict:
    """Classify prose as Chinese or English by CJK share.

    Fenced blocks are stripped first: an ECharts option or a ```map payload is
    JSON with English keys regardless of what language the answer is written
    in, and letting it count would drag every Chinese answer containing a chart
    toward "mixed".
    """
    body = _strip_all_fences(text or "")
    cjk = len(_CJK_RE.findall(body))
    latin = len(_LATIN_WORD_RE.findall(body))
    total = cjk + latin
    if total < 5:
        return LanguageVerdict("unknown", cjk, latin, 0.0)
    ratio = cjk / total
    if ratio >= _ZH_FLOOR:
        lang = "zh"
    elif ratio <= _EN_CEIL:
        lang = "en"
    else:
        lang = "mixed"
    return LanguageVerdict(lang, cjk, latin, ratio)


# ── blocks ──────────────────────────────────────────────────────────────────
_REPORT_RE = re.compile(
    r"<report(?P<attrs>[^>]*)>(?P<body>.*?)</report\s*>", re.S | re.I
)
_REPORT_TITLE_RE = re.compile(r"""title\s*=\s*["'](?P<title>[^"']*)["']""", re.I)
_QUESTION_RE = re.compile(r"<question\s*>(?P<body>.*?)</question\s*>", re.S | re.I)
_TEXTBLOCK_RE = re.compile(
    r"<textblock(?P<attrs>[^>]*)>(?P<body>.*?)</textblock\s*>", re.S | re.I
)
_TB_TYPE_RE = re.compile(r"""type\s*=\s*["'](?P<v>[^"']*)["']""", re.I)
_TB_SUBJECT_RE = re.compile(r"""subject\s*=\s*["'](?P<v>[^"']*)["']""", re.I)
# Decoration SYSTEM_PROMPT bans inside the block: the deliverable is "plain
# finished text only". `**bold**`, `#` headings, and `[n]` citation markers.
_TB_DECORATION_RE = re.compile(r"\*\*[^*\n]+\*\*|^\s{0,3}#{1,6}\s+\S|\[\d{1,3}\]", re.M)


@dataclass
class Report:
    title: str | None
    body: str
    start: int
    end: int

    @property
    def words(self) -> int:
        return count_words(self.body)


def extract_reports(text: str) -> list[Report]:
    out = []
    for m in _REPORT_RE.finditer(text or ""):
        tm = _REPORT_TITLE_RE.search(m.group("attrs") or "")
        out.append(
            Report(
                title=tm.group("title").strip() if tm else None,
                body=m.group("body").strip(),
                start=m.start(),
                end=m.end(),
            )
        )
    return out


def extract_fences(text: str, lang: str) -> list[str]:
    """Return the bodies of every ```<lang> fenced block.

    Written as a scanner rather than one regex because charts legitimately
    appear *inside* a report body that itself may contain other fences, and a
    non-greedy regex across nested content picks the wrong closing fence. Also
    tolerates the 4-backtick form the mapping skill's own docs use.
    """
    out: list[str] = []
    if not text:
        return out
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(?P<ticks>`{3,})\s*(?P<lang>[A-Za-z0-9_-]+)\s*$", lines[i])
        if not m or m.group("lang").lower() != lang.lower():
            i += 1
            continue
        ticks = m.group("ticks")
        body: list[str] = []
        i += 1
        while i < len(lines) and not re.match(rf"^{ticks}\s*$", lines[i]):
            body.append(lines[i])
            i += 1
        i += 1  # step over the closing fence
        out.append("\n".join(body))
    return out


# Fences that are some other skill's deliverable rather than code the user
# asked to be written. `charts_valid` and `map_fence` already grade these, and
# counting them as "the model produced code" would let a chart answer satisfy a
# code-writing check.
_NON_CODE_FENCE_LANGS = {"echarts", "map"}


def extract_code_fences(text: str, lang: str | None = None) -> list[tuple[str, str]]:
    """Every fenced code block as `(lang, body)`, chart and map fences excluded.

    Same scanner shape as `extract_fences` — see its docstring for why this is
    not one regex. Differs in that the language is an output rather than a
    filter, because the question a code check asks is "did it produce code in a
    fence at all", and a model writing Python under ```py or no tag at all has
    still done the thing being asked.

    An untagged fence counts, with `lang` reported as "". That is deliberate:
    the alternative is scoring a correct answer down for a missing tag, which
    the prompt does not require.
    """
    out: list[tuple[str, str]] = []
    if not text:
        return out
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(?P<ticks>`{3,})\s*(?P<lang>[A-Za-z0-9_+-]*)\s*$", lines[i])
        if not m:
            i += 1
            continue
        found = m.group("lang").lower()
        ticks = m.group("ticks")
        body: list[str] = []
        i += 1
        while i < len(lines) and not re.match(rf"^{ticks}\s*$", lines[i]):
            body.append(lines[i])
            i += 1
        i += 1
        if found in _NON_CODE_FENCE_LANGS:
            continue
        if lang is not None and found != lang.lower():
            continue
        out.append((found, "\n".join(body)))
    return out


@dataclass
class ParsedBlock:
    raw: str
    data: Any | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def _parse_json_block(raw: str) -> ParsedBlock:
    try:
        return ParsedBlock(raw=raw, data=json.loads(raw))
    except json.JSONDecodeError as e:
        return ParsedBlock(raw=raw, data=None, error=f"invalid JSON: {e}")


def extract_charts(text: str) -> list[ParsedBlock]:
    return [_parse_json_block(b) for b in extract_fences(text, "echarts")]


def extract_maps(text: str) -> list[ParsedBlock]:
    return [_parse_json_block(b) for b in extract_fences(text, "map")]


@dataclass
class QuestionBlock:
    parsed: ParsedBlock
    is_last: bool
    trailing: str


@dataclass
class TextBlock:
    type: str | None
    subject: str | None
    body: str
    decoration: list[str]

    @property
    def words(self) -> int:
        return count_words(self.body)


def extract_textblocks(text: str) -> list[TextBlock]:
    """`<textblock>…</textblock>` deliverables, per SYSTEM_PROMPT's writing rules.

    This is the writing path's output contract — a rewrite, translation, draft
    or email goes inside the block and nowhere else. It replaces the `---`
    horizontal rules that `has_delimiters` still looks for; that convention was
    retired from the prompt (which now says "Use headers instead of horizontal
    rules"), so the old check scored a model *down* for obeying the current
    instructions.

    `decoration` collects the markdown the prompt forbids inside the block
    (`**bold**`, `#` headings, `[n]` markers) — the block is meant to hold
    finished plain text a user can paste elsewhere, so formatting leaking in
    is a real defect rather than a style quibble.
    """
    out: list[TextBlock] = []
    for m in _TEXTBLOCK_RE.finditer(text or ""):
        attrs = m.group("attrs") or ""
        body = m.group("body").strip()
        tm = _TB_TYPE_RE.search(attrs)
        sm = _TB_SUBJECT_RE.search(attrs)
        out.append(
            TextBlock(
                type=tm.group("v").strip() if tm else None,
                subject=sm.group("v").strip() if sm else None,
                body=body,
                decoration=[d.strip() for d in _TB_DECORATION_RE.findall(body)][:3],
            )
        )
    return out


def extract_question(text: str) -> QuestionBlock | None:
    """The ask-question skill requires the block to be the last thing in the
    message — the frontend renders it as a form and anything after it is
    unreachable — so `is_last` is part of the parse, not a separate check."""
    matches = list(_QUESTION_RE.finditer(text or ""))
    if not matches:
        return None
    m = matches[-1]
    trailing = (text[m.end():] or "").strip()
    return QuestionBlock(
        parsed=_parse_json_block(m.group("body").strip()),
        is_last=not trailing,
        trailing=trailing[:200],
    )


# ── citations ───────────────────────────────────────────────────────────────
_CITE_RE = re.compile(r"\[(\d{1,3})\]")
# Full-width / dagger-suffixed forms the prompt explicitly bans.
_BAD_CITE_RE = re.compile(r"[【［]\s*\d+\s*[†‡]?[^】］]*[】］]")
# A [n] still followed by prose *on the same line* is mid-sentence, not the
# end-of-paragraph cluster the prompt mandates.
#
# The lookahead matches horizontal whitespace only. It used to be `\s*`, which
# also matches newlines, so the correctly-placed cluster in
#
#     海狮属于有耳海豹科。[4][6]
#
#     日常语言中的"海豹"有时会泛指所有鳍足类。
#
# was reported as mid-sentence: `[6]`, then the blank line, then the first
# character of the *next* paragraph. Every frontier model tripped it on most
# cases (5 of 9 for gpt-5.6-luna), so the check was measuring its own bug
# rather than citation placement. A cluster that ends a line is exactly what
# the prompt asks for; only trailing prose on the same line is a violation.
_MIDSENTENCE_CITE_RE = re.compile(
    r"\[\d{1,3}\](?=[^\S\n]*[A-Za-z0-9一-鿿])"
)


@dataclass
class Citations:
    numbers: list[int] = field(default_factory=list)
    bad_format: list[str] = field(default_factory=list)
    midsentence: list[str] = field(default_factory=list)

    @property
    def distinct(self) -> set[int]:
        return set(self.numbers)


def find_citations(text: str) -> Citations:
    text = text or ""
    return Citations(
        numbers=[int(n) for n in _CITE_RE.findall(text)],
        bad_format=_BAD_CITE_RE.findall(text)[:5],
        midsentence=[
            text[max(0, m.start() - 30):m.end() + 20]
            for m in _MIDSENTENCE_CITE_RE.finditer(text)
        ][:5],
    )


# ── format-compliance helpers ───────────────────────────────────────────────
_MD_LINK_RE = re.compile(r"\[[^\]]+\]\((?:https?://|www\.)[^)]+\)")
_BARE_URL_RE = re.compile(r"(?<![(\w/])(?:https?://|www\.)[^\s<>()\[\]]+")
# Box-drawing, block and heavy-ASCII glyphs that only ever appear in text art.
_ART_CHARS = set("│─┌┐└┘├┤┬┴┼║═╔╗╚╝╠╣╦╩╬▀▄█▌▐░▒▓▲▼◄►↑↓←→")
# The subset that is damning on its own *outside* a fence. Arrows are excluded:
# `→` is ordinary prose punctuation ("输入 → 处理 → 输出"), and three bullets in
# a row each carrying one would otherwise be reported as a drawing.
_BOX_CHARS = _ART_CHARS - set("▲▼◄►↑↓←→")


def find_hyperlinks(text: str) -> list[str]:
    """Markdown links and bare URLs, excluding anything inside a code fence.

    Code fences are stripped first because an ```echarts option or a
    ```map block legitimately contains URLs in its data, and a chart's JSON is
    not prose the "no hyperlinks" rule is about.
    """
    stripped = _strip_all_fences(text)
    return (_MD_LINK_RE.findall(stripped) + _BARE_URL_RE.findall(stripped))[:5]


# Punctuation a hand-drawn diagram is built out of. Deliberately excludes `/`
# and `\`, which diagrams do use but which also carry LaTeX (`\frac`, `\[ \]`)
# — a false positive there would fail a legitimately formatted formula.
_ART_LINE_CHARS = set("|+-_=<>^v*.")


def _is_art_line(line: str) -> bool:
    """One line that looks drawn rather than written.

    Box-drawing glyphs settle it outright. Otherwise the line has to be *mostly*
    diagram punctuation: a Markdown table row (`| Client | Server |`) is ~16%
    and stays prose, while a sequence-diagram arrow (`|------ SYN ----->|`) is
    ~80% and does not. The 4-character floor keeps `A -> B` in ordinary prose
    from qualifying.
    """
    s = line.strip()
    if len(s) < 4:
        return False
    if any(ch in _BOX_CHARS for ch in s):
        return True
    n = sum(1 for ch in s if ch in _ART_LINE_CHARS)
    return n >= 4 and n / len(s) >= 0.4


def find_ascii_art(text: str) -> list[str]:
    """Drawings rather than code — inside fences and out.

    A fenced block counts as art when box-drawing glyphs appear, or when its
    lines are overwhelmingly made of `|`, `+`, `-` and spaces — the shape a
    hand-drawn table or flowchart takes. Language-tagged fences we know to be
    data (echarts / map) and real code are exempt.

    Unfenced prose is scanned too, because nothing requires a model to fence a
    drawing: gpt-5.6-luna answered the TCP-handshake case with a bare
    three-line sequence diagram, which the `right_sized` judge objected to
    while this check passed it. There the bar is a *run* of three consecutive
    drawn lines, not a single one — a `---` rule and a Markdown table's
    `|---|---|` separator are each one line and stay legal.
    """
    out = []
    for raw, lang in _iter_fences(text):
        if lang.lower() in {"echarts", "map", "json", "python", "js", "ts", "bash", "sql"}:
            continue
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            continue
        if any(ch in _ART_CHARS for ch in raw):
            out.append(raw[:200])
            continue
        arty = sum(1 for ln in lines if re.fullmatch(r"[\s|+\-_=<>^v*.]{4,}", ln))
        if len(lines) >= 3 and arty / len(lines) >= 0.6:
            out.append(raw[:200])

    run: list[str] = []
    for line in _strip_all_fences(text).splitlines():
        if _is_art_line(line):
            run.append(line)
            continue
        if len(run) >= 3:
            out.append("\n".join(run)[:200])
        run = []
    if len(run) >= 3:
        out.append("\n".join(run)[:200])
    return out[:3]


_LATEX_DOLLAR_RE = re.compile(r"(?<!\\)\$\$?(?!\s)[^$\n]{1,120}?(?<!\\)\$\$?")
# \(5\) / \(2026\) — a bare number or unit wrapped in math delimiters, which the
# prompt calls out specifically as the wrong reflex.
_LATEX_TRIVIAL_RE = re.compile(r"\\\(\s*[\d.,%\s]{1,12}\s*\\\)")


# Inside a `$…$` span, what makes it actually mathematical: a TeX control
# sequence, a sub/superscript, or a standalone letter used as a variable. The
# lookbehind also rejects digits and `.` so a unit glued to a number — the `B`
# of `$16.7B` — reads as currency rather than as a variable named B.
_MATHY_RE = re.compile(r"\\[A-Za-z]+|[\^_]|(?<![A-Za-z0-9.])[a-zA-Z](?![A-Za-z])")


def _is_currency_pair(span: str) -> bool:
    """`$700–$1,000` and `$16.7B | $26.9B` are two prices, not a math span.

    `_LATEX_DOLLAR_RE` cannot tell the difference — it sees an opening `$`,
    some non-`$` text, and a closing `$` — so every price range in an English
    answer was reported as dollar-delimited LaTeX. That is the *opposite* of
    what SYSTEM_PROMPT asks for: it names `$10` as something that "should just be
    typed as normal text". The check was failing the behaviour it exists to
    enforce, on 3 of 9 cases for gpt-5.6-luna.

    A real math span carries a control sequence, a sub/superscript, or a
    single-letter variable; a price range carries digits, separators and
    unit suffixes. Requiring one mathy token keeps `$\\frac{a}{b}$` (a genuine
    violation — it should be `\\( \\)`) failing while letting money through.

    The word test catches the other shape: two prices with a whole clause
    between them, as in `$193.7 billion in FY2026, while AMD's entire 2025
    revenue was $`. That one slips past the mathy-token test because `AMD's`
    contributes a lone `s`. Inline math is short and symbolic; three or more
    ordinary words means the `$`s bracket prose, not an equation.
    """
    inner = span.strip("$")
    if not inner:
        return False
    words = re.findall(r"[A-Za-z]{3,}", re.sub(r"\\[A-Za-z]+", " ", inner))
    if len(words) >= 3:
        return True
    return not _MATHY_RE.search(inner)


def find_latex_problems(text: str) -> list[str]:
    stripped = _strip_all_fences(text)
    dollar = [m for m in _LATEX_DOLLAR_RE.findall(stripped) if not _is_currency_pair(m)]
    return (
        [f"dollar-delimited: {m}" for m in dollar[:3]]
        + [f"trivial: {m}" for m in _LATEX_TRIVIAL_RE.findall(stripped)[:3]]
    )


_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")


def starts_with_header(text: str) -> bool:
    for line in (text or "").splitlines():
        if line.strip():
            return bool(_HEADER_RE.match(line))
    return False


def ends_with_question(text: str) -> bool:
    """Last non-empty line reads as a question — the follow-up the prompt asks
    for after a writing task. Matches both ASCII and full-width marks."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return bool(lines) and lines[-1].endswith(("?", "？"))


# ── prose extraction ────────────────────────────────────────────────────────
def prose_only(text: str) -> str:
    """The answer with every structured block removed.

    Word counts run on this: a `word_count max` on a chat answer is about how
    much the model *said*, and letting a 400-line ECharts option or an embedded
    report count toward it would make the threshold meaningless.
    """
    out = _REPORT_RE.sub(" ", text or "")
    out = _QUESTION_RE.sub(" ", out)
    # Same reasoning as reports: a `<textblock>` holds the deliverable, not the
    # model's remarks about it, and the prompt asks for exactly one short line
    # of those. Its length is bounded by the `textblock` check's own
    # min_words/max_words instead.
    out = _TEXTBLOCK_RE.sub(" ", out)
    return _strip_all_fences(out)


def conversational_text(text: str) -> str:
    """Everything the assistant *said to* the user — not what it produced *for*
    them.

    In: prose, `<report>` bodies, and `<question>` blocks. `prose_only` strips
    all of those, because it answers "how much did the model say"; language
    detection needs the opposite. Without the question block,
    `response_language` reported "answer too short to classify" on exactly the
    turns where asking a clarifying question is the correct behaviour — all
    four of gpt-5.6-luna's failures on the 28-case set were this, on
    `ask-question/laptop-choice`, `web-research/ambiguous-scope` and both turns
    of `trip-advisor/weekend-nyc`. The check was scoring the model down for
    obeying the ask-question skill.

    Out: `<textblock>` bodies. A report is addressed to the user — a Chinese
    question gets a Chinese report — but a textblock is a deliverable whose
    language the *task* dictates: "把这段翻译成英文" is correctly answered with
    Chinese commentary wrapped around an English block. Counting the block
    scored every cross-language writing task as `mixed`, which mattered more
    than it sounds — `response_language` is in the *gate* of
    `finetune/pro_agent/queries.yaml`, so it silently discarded whole
    trajectories, and translation is the one task where the mismatch is not a
    defect but the entire point. Measured on the first batch of translation
    traces it would have thrown away 3 of 8, including both cases where the
    target language was neither the query's nor Chinese. The 129 collected rows
    before it never tripped this only because none of their write-rewrite
    queries crossed languages.
    """
    parts = [prose_only(text)]
    parts += [r.body for r in extract_reports(text)]
    block = extract_question(text)
    if block and block.parsed.ok and isinstance(block.parsed.data, dict):
        for q in block.parsed.data.get("questions") or []:
            if not isinstance(q, dict):
                continue
            parts.append(str(q.get("prompt") or ""))
            for opt in q.get("options") or []:
                if isinstance(opt, dict):
                    parts.append(str(opt.get("label") or ""))
    return "\n\n".join(p for p in parts if p and p.strip())


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


_FENCE_OPEN_RE = re.compile(r"^(?P<ticks>`{3,})\s*(?P<lang>[A-Za-z0-9_-]*)\s*$")


def _iter_fences(text: str):
    """Yield `(body, lang)` for every fenced block."""
    lines = (text or "").splitlines()
    i = 0
    while i < len(lines):
        m = _FENCE_OPEN_RE.match(lines[i])
        if not m:
            i += 1
            continue
        ticks, lang = m.group("ticks"), m.group("lang") or ""
        body: list[str] = []
        i += 1
        while i < len(lines) and not re.match(rf"^{ticks}\s*$", lines[i]):
            body.append(lines[i])
            i += 1
        i += 1
        yield "\n".join(body), lang


def _strip_all_fences(text: str) -> str:
    lines = (text or "").splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        m = _FENCE_OPEN_RE.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        ticks = m.group("ticks")
        i += 1
        while i < len(lines) and not re.match(rf"^{ticks}\s*$", lines[i]):
            i += 1
        i += 1
    return "\n".join(out)
