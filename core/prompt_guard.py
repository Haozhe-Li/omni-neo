"""Output-side guard: detect verbatim / near-verbatim system prompt leakage.

Detection is fingerprint-only — the guard is built from the *actual* system
prompt texts registered at startup (see `register_sensitive_prompts`,
called from `main.py` with `core.agent.SYSTEM_PROMPTS`), and fires only when
streamed output demonstrably reproduces a stretch of one of them. There are
deliberately no keyword/regex heuristics: topic words like "system prompt",
"reasoning" or "token" appear constantly in benign answers (measured 9/12
false positives on ordinary assistant replies with the previous regex
layer — "let me explain my reasoning", "store your API key in an env var",
"how do I write a good system prompt for my agent?"), so matching on
vocabulary is unusable as a trigger for anything consequential. Matching on
the prompts themselves instead:

- normalizes both sides (lowercase, drop punctuation/markdown, split CJK to
  per-character tokens, strip zero-width/format chars some obfuscation
  attacks insert between letters) so cosmetic reformatting doesn't dodge it;
- fingerprints every word n-gram of every registered prompt into one set;
- slides over the candidate window's n-grams and fires on either signal:
    1. verbatim run — `min_run_tokens` *consecutive* normalized tokens
       shared with a prompt (overlapping n-gram hits chain into runs). A
       dozen-word exact quote of a specific prompt is essentially
       impossible to produce by accident.
    2. dense overlap — the window is long enough, and a large fraction of
       ALL its n-grams appear in a prompt. Catches quote-with-ellipsis
       leaks ("...chopped every 8 words...") whose individual runs stay
       short of the verbatim-run threshold.

Known limitation, by design: a paraphrased or translated leak shares no
surface n-grams and will not fire. This guard's job is to make the cheap
attack (verbatim dump) fail loudly and observably — the durable defense
remains keeping real secrets out of the prompt entirely.
"""

import logging
import re
from typing import Iterable

from langsmith import tracing_context

from core.llm import prompt_guard_llm

logger = logging.getLogger(__name__)

# Zero-width/format chars some obfuscation attacks sprinkle between letters
# to dodge substring matching (e.g. U+200B between every character).
_ZERO_WIDTH_RE = re.compile(r"[​-‏⁠﻿]")
# CJK ideographs get split to one token per character — otherwise a whole
# Chinese/Japanese sentence has no whitespace to split on and collapses into
# a single unsplittable token, so it can never form an n-gram at all.
_CJK_RE = re.compile(r"([㐀-鿿豈-﫿])")
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    text = _ZERO_WIDTH_RE.sub("", text.lower())
    text = _CJK_RE.sub(r" \1 ", text)
    return _TOKEN_RE.findall(text)


class PromptLeakGuard:
    """N-gram fingerprint matcher over the registered sensitive prompts.

    Args:
        prompt_texts: the sensitive texts to fingerprint (system prompts).
        ngram_size: tokens per fingerprint n-gram. 5 is small enough that a
            12-token quote yields 8 chainable hits, large enough that a
            5-gram colliding with ordinary prose is rare.
        min_run_tokens: consecutive shared tokens that count as a verbatim
            quote. 12 ~= a full clause — generic 5-8 word phrases ("you are
            a helpful assistant that") stay safely below it.
        containment_threshold: fraction of the window's n-grams that must be
            prompt n-grams for the dense-overlap signal.
        containment_min_tokens: dense overlap is only meaningful on a window
            with some substance; shorter windows rely on the run signal.
    """

    def __init__(
        self,
        prompt_texts: Iterable[str],
        ngram_size: int = 5,
        min_run_tokens: int = 12,
        containment_threshold: float = 0.35,
        containment_min_tokens: int = 40,
    ):
        self.ngram_size = ngram_size
        self.min_run_tokens = min_run_tokens
        self.containment_threshold = containment_threshold
        self.containment_min_tokens = containment_min_tokens

        self._ngrams: set[str] = set()
        for text in prompt_texts:
            if not isinstance(text, str) or not text.strip():
                continue
            tokens = _tokenize(text)
            for i in range(len(tokens) - ngram_size + 1):
                self._ngrams.add(" ".join(tokens[i : i + ngram_size]))

    @property
    def fingerprint_size(self) -> int:
        return len(self._ngrams)

    def detect(self, text: str) -> tuple[bool, str, float]:
        """Return (leaking, reason, score).

        reason is one of "empty", "too_short", "verbatim_run",
        "dense_overlap", "clean". score is the longest shared token run for
        "verbatim_run"/"clean", or the n-gram containment ratio for
        "dense_overlap".
        """
        if not text or not isinstance(text, str):
            return False, "empty", 0.0
        if not self._ngrams:
            return False, "empty", 0.0

        tokens = _tokenize(text)
        n = self.ngram_size
        if len(tokens) < n:
            return False, "too_short", 0.0

        total = len(tokens) - n + 1
        hits = 0
        best_run = 0
        current_run = 0
        for i in range(total):
            if " ".join(tokens[i : i + n]) in self._ngrams:
                hits += 1
                current_run += 1
                if current_run > best_run:
                    best_run = current_run
            else:
                current_run = 0

        # k consecutive n-gram hits => a shared run of k + n - 1 tokens.
        run_tokens = best_run + n - 1 if best_run else 0
        if run_tokens >= self.min_run_tokens:
            return True, "verbatim_run", float(run_tokens)

        if len(tokens) >= self.containment_min_tokens:
            containment = hits / total
            if containment >= self.containment_threshold:
                return True, "dense_overlap", containment

        return False, "clean", float(run_tokens)


def build_prompt_leakage_guard(
    prompt_texts: Iterable[str],
    ngram_size: int = 5,
    min_run_tokens: int = 12,
    containment_threshold: float = 0.35,
    containment_min_tokens: int = 40,
) -> PromptLeakGuard:
    return PromptLeakGuard(
        prompt_texts=prompt_texts,
        ngram_size=ngram_size,
        min_run_tokens=min_run_tokens,
        containment_threshold=containment_threshold,
        containment_min_tokens=containment_min_tokens,
    )


_REGISTERED_SENSITIVE_PROMPTS: list[str] = []
_DEFAULT_LEAK_GUARD = build_prompt_leakage_guard([])


def register_sensitive_prompts(prompt_texts: Iterable[str]) -> None:
    global _REGISTERED_SENSITIVE_PROMPTS
    global _DEFAULT_LEAK_GUARD
    seen = set(_REGISTERED_SENSITIVE_PROMPTS)
    for text in prompt_texts:
        if isinstance(text, str) and text.strip() and text not in seen:
            _REGISTERED_SENSITIVE_PROMPTS.append(text)
            seen.add(text)
    _DEFAULT_LEAK_GUARD = build_prompt_leakage_guard(_REGISTERED_SENSITIVE_PROMPTS)
    logger.info(
        "[prompt_guard] fingerprinted %d prompt(s), %d n-grams",
        len(_REGISTERED_SENSITIVE_PROMPTS),
        _DEFAULT_LEAK_GUARD.fingerprint_size,
    )


def has_prompt_leakage(text: str) -> bool:
    detected, reason, score = _DEFAULT_LEAK_GUARD.detect(text)
    if detected:
        logger.warning("[prompt_guard] leakage detected: reason=%s score=%.2f", reason, score)
    return detected


def sanitize_output_text(
    text: str,
    fallback: str = "I’m sorry, but I can’t share that.",
) -> tuple[str, bool]:
    if has_prompt_leakage(text):
        return fallback, True
    return text, False


async def is_harmful(query: str) -> bool:
    if len(query) > 50:
        return False
    try:
        messages = [
            (
                "human",
                query,
            ),
        ]
        with tracing_context(project_name="prompt-guard"):
            res = float((await prompt_guard_llm.ainvoke(messages)).content)
        return res > 0.5
    except Exception as e:
        logger.warning(f"[prompt_guard] is_harmful check failed, allowing through: {e}")
        return False


# if __name__ == "__main__":
#     print(is_harmful("What is the capital of France?"))
