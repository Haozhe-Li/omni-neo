"""Discover the chat models to evaluate by reflecting over `core/llm.py`.

Deliberately reflective rather than a hand-copied list: a model added to
`core/llm.py` is then covered by the eval automatically instead of being
silently skipped until someone remembers to update a second list. The only
hand-maintained part is the exclusion set below, which is short and explicit.

Labels are derived from the *variable name*, not the model id, because the same
model id appears under several variables that differ in ways the id doesn't
capture — `gpt-oss-120b` is six separate entries here (three reasoning efforts
times two providers) and they must not collapse into one label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

import core.llm as llm_module

# Models excluded from the eval roster, with the reason each is out. Exclusion
# is scoped to the eval only — every one of these is still constructed in
# core/llm.py and, in llama's case, still serves production traffic.
_EXCLUDED: dict[str, str] = {
    # An 86M sequence classifier that scores a span for injection intent, not a
    # conversational model. Handing it a pro system prompt and a research task
    # is meaningless, not merely a bad score.
    "prompt_guard_2_86m": "not a chat model (injection classifier)",
    # Measured at the pro profile and found below the floor: it fails to emit
    # well-formed tool calls at all ("Failed to call a function" on 4 of 18
    # smoke cases), so its scores describe a parse failure rather than any
    # capability worth tracking. Still the autocomplete model in production
    # (core/auto_complete.py) — a one-shot structured-output call it handles
    # fine — so it stays in core/llm.py.
    "llama3_1_8b": "below the pro-profile floor; still used for autocomplete",
    # A LoRA fine-tuned to emit widget-routing JSON and nothing else. Handed a
    # pro system prompt and a research task it would answer with
    # {"widgets":[]}, so its score would describe the fine-tune rather than any
    # chat capability. finetune/widget_predictor/evaluate.py scores it instead,
    # against its own held-out split.
    "omni_widget_predictor_14b": "task-specific widget classifier, not a chat model",
}

_PROVIDER_BY_CLASS = {
    "ChatCerebras": "cerebras",
    "ChatGroq": "groq",
    "ChatGoogleGenerativeAI": "google_genai",
    "ChatOpenAI": "openai",
    "ChatAnthropic": "anthropic",
}


@dataclass(frozen=True)
class ModelSpec:
    var_name: str
    label: str
    model_id: str
    provider: str
    family: str
    reasoning_effort: str | None
    llm: BaseChatModel

    def __str__(self) -> str:
        return self.label


def _provider_of(llm: BaseChatModel) -> str:
    cls = type(llm).__name__
    if cls in _PROVIDER_BY_CLASS:
        return _PROVIDER_BY_CLASS[cls]
    # core/llm.py subclasses ChatCerebras to patch reasoning extraction; walk
    # the MRO so the subclass still resolves to its provider.
    for base in type(llm).__mro__:
        if base.__name__ in _PROVIDER_BY_CLASS:
            return _PROVIDER_BY_CLASS[base.__name__]
    return cls.lower()


def _model_id_of(llm: BaseChatModel) -> str:
    for attr in ("model_name", "model", "model_id"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def _family_of(model_id: str) -> str:
    """Strip provider routing prefixes so the same weights group together.

    Groq serves gpt-oss as `openai/gpt-oss-120b` and Cerebras as plain
    `gpt-oss-120b`; without this they'd land in different families and the 2x3
    provider-vs-effort grid would never line up.
    """
    return model_id.split("/")[-1]


# Display-only relabeling, keyed by var_name — does NOT touch `family`
# (still derived from the real model_id, so pricing.yaml lookups and the
# gpt-oss-120b provider grid are untouched) or the variable itself in
# core/llm.py (still named `gemini_flash_lite_latest` there; nothing wired to
# a production role gets renamed by this).
#
# `gemini-flash-lite-latest` is a rolling alias but currently resolves to an
# old snapshot rather than Google's actual latest Flash-Lite — labeling it
# "-latest" in the dashboard/leaderboard overstates its recency.
_LABEL_OVERRIDES: dict[str, str] = {
    "gemini_flash_lite_latest": "gemini-flash-lite-3-5",
}


def _label_of(var_name: str) -> str:
    return _LABEL_OVERRIDES.get(var_name, var_name.replace("_", "-"))


def discover_models() -> list[ModelSpec]:
    """Every chat model defined in `core/llm.py`, in definition order.

    Deduplicated by object identity: `core/llm.py` binds role aliases
    (`fast_llm`, `pro_llm`, `update_memories_llm`, ...) to the same instances,
    and evaluating `gpt-oss-120b-low` a second time under the name `fast-llm`
    would just double the bill and split its results across two labels. The
    canonical (first-defined) name wins.
    """
    seen: dict[int, ModelSpec] = {}
    # Exclusion has to propagate by identity, not just by name: core/llm.py
    # binds `prompt_guard_llm = prompt_guard_2_86m`, and skipping only the
    # excluded *name* lets the same object back in under its alias.
    excluded_ids: set[int] = set()
    out: list[ModelSpec] = []
    for var_name, obj in vars(llm_module).items():
        if var_name.startswith("_"):
            continue
        if not isinstance(obj, BaseChatModel):
            continue
        if var_name in _EXCLUDED:
            excluded_ids.add(id(obj))
            continue
        if id(obj) in excluded_ids or id(obj) in seen:
            continue
        model_id = _model_id_of(obj)
        spec = ModelSpec(
            var_name=var_name,
            label=_label_of(var_name),
            model_id=model_id,
            provider=_provider_of(obj),
            family=_family_of(model_id),
            reasoning_effort=getattr(obj, "reasoning_effort", None) or None,
            llm=obj,
        )
        seen[id(obj)] = spec
        out.append(spec)
    return out


def resolve_models(labels: list[str] | None) -> list[ModelSpec]:
    """Select by label, or return everything when no filter is given."""
    available = discover_models()
    if not labels:
        return available
    by_label = {m.label: m for m in available}
    # Also accept the raw variable name, since that is what core/llm.py calls it.
    by_label.update({m.var_name: m for m in available})
    missing = [l for l in labels if l not in by_label]
    if missing:
        raise SystemExit(
            f"unknown model(s): {', '.join(missing)}\n"
            f"available: {', '.join(sorted(m.label for m in available))}"
        )
    picked: list[ModelSpec] = []
    for label in labels:
        spec = by_label[label]
        if spec not in picked:
            picked.append(spec)
    return picked


def group_key(spec: ModelSpec) -> str:
    """`family/provider/effort` — the coordinates of the 2x3 grid."""
    return f"{spec.family}/{spec.provider}/{spec.reasoning_effort or 'default'}"


if __name__ == "__main__":  # `python -m evals.models` to eyeball the matrix
    rows = discover_models()
    width = max(len(m.label) for m in rows)
    print(f"{len(rows)} chat models\n")
    for name, why in sorted(_EXCLUDED.items()):
        print(f"  excluded: {name} — {why}")
    print()
    for m in rows:
        effort = m.reasoning_effort or "-"
        print(f"  {m.label:<{width}}  {m.provider:<13} {m.family:<24} effort={effort}")
