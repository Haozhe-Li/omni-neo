"""Trace and metric structures — what one agent run produced, and what it cost.

A `TurnTrace` is everything observable about one assistant turn: the text, the
ordered tool calls with their results, and the layer-C metrics. `RunTrace`
aggregates the turns of a (possibly multi-turn) case.

Token accounting is the subtle part; see `add_usage`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

SKILL_PATH_RE = re.compile(r"^/skills/(?P<name>[^/]+)/SKILL\.md$")


@dataclass
class ToolCallRecord:
    index: int
    name: str
    args: dict[str, Any]
    result_head: str = ""
    result_full: str = ""

    @property
    def skill_read(self) -> str | None:
        """The skill name when this call is deepagents' progressive-disclosure
        read, else None.

        This is how skill triggering is detected: `SkillsMiddleware` surfaces
        only a skill's name and description in the system prompt and tells the
        model to `read_file` the full SKILL.md on demand, so "did it load the
        skill" is a literal tool call in the trace — no inference needed.
        """
        if self.name != "read_file":
            return None
        path = str(self.args.get("file_path") or self.args.get("path") or "")
        m = SKILL_PATH_RE.match(path)
        return m.group("name") if m else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "i": self.index,
            "name": self.name,
            "args": _truncate_args(self.args),
            "result_head": self.result_head[:600],
        }


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    peak_context_tokens: int = 0

    def add_usage(self, meta: dict | None) -> None:
        """Accumulate one LLM call's `usage_metadata`.

        Summing (rather than taking the last value) is the only rule that is
        correct for all three providers, and the difference is not cosmetic:

        - Cerebras and Groq emit usage once, on the final chunk, as a total.
        - Gemini emits per-chunk *deltas* — `input_tokens` only on the first
          chunk, `output_tokens` spread across chunks — and its final chunk is
          all zeros.

        So "read the last non-empty usage_metadata" silently reports zero
        tokens, and therefore zero cost, for every Gemini model. Summing gives
        the right answer under both conventions, because the single total is
        just a sum with one term. Verified empirically against all three.
        """
        if not meta:
            return
        self.input_tokens += int(meta.get("input_tokens") or 0)
        self.output_tokens += int(meta.get("output_tokens") or 0)
        details_in = meta.get("input_token_details") or {}
        details_out = meta.get("output_token_details") or {}
        self.cached_input_tokens += int(details_in.get("cache_read") or 0)
        self.reasoning_tokens += int(details_out.get("reasoning") or 0)
        self.peak_context_tokens = max(
            self.peak_context_tokens, int(meta.get("input_tokens") or 0)
        )

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "peak_context_tokens": self.peak_context_tokens,
        }


@dataclass
class TurnTrace:
    index: int
    query: str
    text: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    n_llm_turns: int = 0

    # Layer C. Three separate first-token measures because in an agent run they
    # are an order of magnitude apart and answer different questions: `ttft_ms`
    # is when the user sees *something* happen, `ttft_answer_ms` is when prose
    # starts (after every tool call), `ttft_report_ms` is when the reader pane
    # can open.
    ttft_ms: int | None = None
    ttft_answer_ms: int | None = None
    ttft_report_ms: int | None = None
    latency_ms: int = 0
    usage: Usage = field(default_factory=Usage)
    hit_run_limit: bool = False
    error: str | None = None

    # Assistant messages that carried BOTH prose and tool calls, which
    # PRO_PROMPT forbids ("a turn is 100% tool calls or 100% final text"). This
    # is what produces "Let me look that up…" narration leaking into the stream
    # ahead of the real answer, so the offending text is kept for evidence.
    mixed_messages: list[str] = field(default_factory=list)

    def tools_named(self, name: str) -> list[ToolCallRecord]:
        return [t for t in self.tool_calls if t.name == name]

    @property
    def skills_loaded(self) -> list[str]:
        """Skill names in load order, deduplicated."""
        out: list[str] = []
        for call in self.tool_calls:
            skill = call.skill_read
            if skill and skill not in out:
                out.append(skill)
        return out

    def skill_load_index(self, skill: str) -> int | None:
        for call in self.tool_calls:
            if call.skill_read == skill:
                return call.index
        return None

    @property
    def n_searches(self) -> int:
        return len(self.tools_named("google_search"))

    @property
    def n_pages_read(self) -> int:
        return len(self.tools_named("load_web_page"))

    @property
    def distinct_domains(self) -> set[str]:
        out = set()
        for call in self.tools_named("load_web_page"):
            url = str(call.args.get("url") or "")
            host = urlparse(url).netloc.lower()
            if host:
                out.add(host[4:] if host.startswith("www.") else host)
        return out

    def distinct_queries(self, tool: str) -> set[str]:
        """Normalised query strings, so re-running one search three times
        counts once. Without this, a model that pads its tool count with
        repeats scores the same as one that genuinely covered three angles."""
        out = set()
        for call in self.tools_named(tool):
            q = str(call.args.get("query") or "").strip().lower()
            if q:
                out.add(re.sub(r"\s+", " ", q))
        return out


@dataclass
class RunTrace:
    case_id: str
    model_label: str
    turns: list[TurnTrace] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    status: str = "ok"          # ok | error | timeout
    error: str | None = None
    cache_stats: dict[str, int] = field(default_factory=dict)

    @property
    def last(self) -> TurnTrace:
        return self.turns[-1]

    def select(self, turn: int | str) -> list[TurnTrace]:
        """Resolve a check's `turn` selector to the turns it grades."""
        if turn == "last":
            return [self.turns[-1]] if self.turns else []
        if turn in ("any", "all"):
            return list(self.turns)
        if isinstance(turn, int) and 0 <= turn < len(self.turns):
            return [self.turns[turn]]
        return []

    # ── aggregate metrics ───────────────────────────────────────────────────
    @property
    def usage(self) -> Usage:
        total = Usage()
        for t in self.turns:
            total.input_tokens += t.usage.input_tokens
            total.output_tokens += t.usage.output_tokens
            total.cached_input_tokens += t.usage.cached_input_tokens
            total.reasoning_tokens += t.usage.reasoning_tokens
            total.peak_context_tokens = max(
                total.peak_context_tokens, t.usage.peak_context_tokens
            )
        return total

    @property
    def latency_ms(self) -> int:
        return sum(t.latency_ms for t in self.turns)

    @property
    def per_turn_latency_ms(self) -> list[int]:
        return [t.latency_ms for t in self.turns]

    @property
    def n_tool_calls(self) -> int:
        return sum(len(t.tool_calls) for t in self.turns)

    @property
    def n_llm_turns(self) -> int:
        return sum(t.n_llm_turns for t in self.turns)

    @property
    def hit_run_limit(self) -> bool:
        return any(t.hit_run_limit for t in self.turns)

    @property
    def skills_loaded(self) -> list[str]:
        out: list[str] = []
        for t in self.turns:
            for s in t.skills_loaded:
                if s not in out:
                    out.append(s)
        return out

    def citation_numbers(self) -> set[int]:
        return {int(c["n"]) for c in self.citations if c.get("n") is not None}

    def source_text_for(self, n: int) -> str:
        """Full tool output backing citation `n`, for the grounding judge.

        The registry's own `content` is a truncated snippet (a search result
        blurb, or a document's first 1000 characters). Judging whether a source
        supports a claim against a snippet would flag correct citations as
        unsupported whenever the evidence sat past the cutoff — so this returns
        what the model actually read: the full ToolMessage body.
        """
        record = next((c for c in self.citations if c.get("n") == n), None)
        if record is None:
            return ""
        url = record.get("url") or ""
        for turn in self.turns:
            for call in turn.tool_calls:
                if url and url in call.result_full:
                    return call.result_full
        return record.get("content", "")

    def as_trace_json(self) -> list[dict]:
        out = []
        for turn in self.turns:
            for call in turn.tool_calls:
                row = call.as_dict()
                row["turn"] = turn.index
                out.append(row)
        return out


def _truncate_args(args: dict[str, Any], limit: int = 400) -> dict[str, Any]:
    out = {}
    for k, v in (args or {}).items():
        s = v if isinstance(v, (int, float, bool, type(None))) else str(v)
        if isinstance(s, str) and len(s) > limit:
            s = s[:limit] + f"…(+{len(s) - limit})"
        out[k] = s
    return out
