"""Cost calculation, driven by `evals/pricing.yaml`.

This file is the pricing oracle for `eval_results.cost_usd` — the YAML gets
read directly at compute time, so updating a price is a one-line YAML edit
followed by a re-run, never a code change.

Prompt caching is not modelled — every input token bills at list `input` rate.
See `compute_cost` for why the previous blended-cache-ratio formula was
removed. The one assumption left is the OpenAI long-context threshold, flagged
in the YAML header.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

DEFAULT_PRICING_PATH = os.path.join(os.path.dirname(__file__), "pricing.yaml")


@dataclass(frozen=True)
class PriceTier:
    """USD per 1M tokens for one billing tier."""

    input: float
    output: float
    cached_input: float | None = None
    cache_writes: float | None = None  # recorded, never billed — see YAML header


@dataclass(frozen=True)
class ModelPrice:
    provider: str
    family: str
    # {"single": tier} for a flat-rate provider, or {"short_context": tier,
    # "long_context": tier} for OpenAI's tiered models.
    tiers: dict[str, PriceTier]
    speed_tokens_per_s: float | None = None

    @property
    def is_tiered(self) -> bool:
        return "single" not in self.tiers


@dataclass(frozen=True)
class PricingTable:
    version: int
    cache_hit_ratio: float
    long_context_threshold: int
    models: dict[tuple[str, str], ModelPrice]  # (provider, family) -> price

    def get(self, provider: str, family: str) -> ModelPrice | None:
        return self.models.get((provider, family))


def _parse_tier(raw: dict | None) -> PriceTier | None:
    if not raw:
        return None
    return PriceTier(
        input=float(raw["input"]),
        output=float(raw["output"]),
        cached_input=float(raw["cached_input"]) if raw.get("cached_input") is not None else None,
        cache_writes=float(raw["cache_writes"]) if raw.get("cache_writes") is not None else None,
    )


_cache: PricingTable | None = None


def load_pricing(path: str = DEFAULT_PRICING_PATH) -> PricingTable:
    """Parse pricing.yaml. Cached per process — the CLI is a one-shot run, so
    there is no need to watch the file for changes mid-run."""
    global _cache
    if _cache is not None:
        return _cache

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    models: dict[tuple[str, str], ModelPrice] = {}
    for provider, provider_block in (raw.get("providers") or {}).items():
        for family, entry in (provider_block.get("models") or {}).items():
            if "short_context" in entry or "long_context" in entry:
                tiers = {
                    tier_name: t
                    for tier_name in ("short_context", "long_context")
                    if (t := _parse_tier(entry.get(tier_name))) is not None
                }
            else:
                tiers = {"single": _parse_tier(entry)}
            models[(provider, family)] = ModelPrice(
                provider=provider,
                family=family,
                tiers=tiers,
                speed_tokens_per_s=entry.get("speed_tokens_per_s"),
            )

    assumptions = raw.get("assumptions") or {}
    _cache = PricingTable(
        version=int(raw.get("version", 1)),
        cache_hit_ratio=float(assumptions.get("cache_hit_ratio", 0.0)),
        long_context_threshold=int(assumptions.get("openai_long_context_threshold_tokens", 1 << 62)),
        models=models,
    )
    return _cache


def _select_tier(price: ModelPrice, table: PricingTable, peak_context_tokens: int) -> PriceTier:
    if not price.is_tiered:
        return price.tiers["single"]
    # peak_context_tokens is the largest SINGLE call's input in the run, not the
    # summed total — the closest proxy available for "which tier would this
    # request have billed at", since usage is only tracked in aggregate across
    # a run's many LLM calls, not per call. A run whose peak call crossed the
    # threshold is billed as long-context for its whole cost, which over-bills
    # any earlier, smaller calls in the same run — a documented approximation,
    # not a precise per-call tier lookup.
    if peak_context_tokens >= table.long_context_threshold and "long_context" in price.tiers:
        return price.tiers["long_context"]
    return price.tiers.get("short_context") or next(iter(price.tiers.values()))


def compute_cost(
    provider: str,
    family: str,
    *,
    input_tokens: int,
    output_tokens: int,
    peak_context_tokens: int = 0,
    table: PricingTable | None = None,
) -> float | None:
    """USD for one run's total usage, or None when the model has no price on
    file — never 0.0, so an unpriced model reads as "unknown" in the
    dashboard rather than winning every cost comparison it appears in.

    Every input token is billed at the plain `input` rate. Prompt caching is
    deliberately NOT modelled: the previous formula blended a flat, assumed
    75% cache-hit ratio into an effective input price, which made every cost
    figure a function of a number nobody measured. Providers do not report cache
    hits comparably (Cerebras and Gemini expose `cache_read`, Groq exposes
    nothing), so the assumption could never be validated — and it moved costs by
    up to 10x on models with a steep cached discount. Charging list price for
    everything is wrong in a known direction and by a knowable amount, which is
    more useful than being wrong by an unknown one.

    `cached_input` stays in `pricing.yaml` as recorded reference data; nothing
    reads it for billing.
    """
    table = table or load_pricing()
    price = table.get(provider, family)
    if price is None:
        return None
    tier = _select_tier(price, table, peak_context_tokens)

    cost = (input_tokens / 1_000_000) * tier.input + (output_tokens / 1_000_000) * tier.output
    return round(cost, 6)


if __name__ == "__main__":  # `python -m evals.pricing` to eyeball the table
    t = load_pricing()
    print(f"pricing.yaml version={t.version}  "
          f"long_context_threshold={t.long_context_threshold}  (cache not modelled)\n")
    for (provider, family), price in sorted(t.models.items()):
        for tier_name, tier in price.tiers.items():
            print(f"  {provider:<10} {family:<18} {tier_name:<14} "
                  f"in=${tier.input:<7} out=${tier.output:<7}   "
                  f"(cached=${str(tier.cached_input)} — recorded, not billed)")
