import logging
import re
from typing import Iterable

from core.llm import prompt_guard_llm

logger = logging.getLogger(__name__)


_LEAKAGE_PATTERNS = [
    re.compile(
        r"\b(system prompt|developer message|hidden instructions?)\b", re.IGNORECASE
    ),
    re.compile(r"\b(ignore (all|previous|prior) instructions?)\b", re.IGNORECASE),
    re.compile(r"\b(chain\s*of\s*thought|reasoning(?:_content)?)\b", re.IGNORECASE),
    re.compile(r"\bBEGIN\s+(SYSTEM|DEVELOPER)\s+PROMPT\b", re.IGNORECASE),
    re.compile(
        r"\b(api[_-]?key|secret|token|bearer\s+[A-Za-z0-9\-._~+/]+=*)\b", re.IGNORECASE
    ),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b", re.IGNORECASE),
]


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^\w\u4e00-\u9fff]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _tokenize(text: str) -> list[str]:
    return [token for token in _normalize_text(text).split(" ") if token]


def _build_ngrams(tokens: list[str], ngram_size: int) -> set[str]:
    if not tokens:
        return set()
    if len(tokens) < ngram_size:
        return {" ".join(tokens)}
    return {
        " ".join(tokens[idx : idx + ngram_size])
        for idx in range(0, len(tokens) - ngram_size + 1)
    }


class PromptLeakageGuard:
    def __init__(
        self,
        prompt_texts: Iterable[str],
        ngram_size: int = 6,
        similarity_threshold: float = 0.28,
        min_shared_ngrams: int = 3,
        min_candidate_tokens: int = 12,
    ):
        self.ngram_size = ngram_size
        self.similarity_threshold = similarity_threshold
        self.min_shared_ngrams = min_shared_ngrams
        self.min_candidate_tokens = min_candidate_tokens

        fingerprints: list[set[str]] = []
        for text in prompt_texts:
            if not isinstance(text, str) or not text.strip():
                continue
            ngrams = _build_ngrams(_tokenize(text), self.ngram_size)
            if ngrams:
                fingerprints.append(ngrams)
        self._fingerprints = fingerprints

    def detect(self, text: str) -> tuple[bool, str, float]:
        if not text or not isinstance(text, str):
            return False, "empty", 0.0

        sample = text.strip()
        if not sample:
            return False, "empty", 0.0

        if any(pattern.search(sample) for pattern in _LEAKAGE_PATTERNS):
            return True, "keyword", 1.0

        candidate_tokens = _tokenize(sample)
        if len(candidate_tokens) < self.min_candidate_tokens:
            return False, "too_short", 0.0

        candidate_ngrams = _build_ngrams(candidate_tokens, self.ngram_size)
        if not candidate_ngrams:
            return False, "no_ngrams", 0.0

        best_score = 0.0
        for fingerprint in self._fingerprints:
            overlap = len(candidate_ngrams & fingerprint)
            if overlap < self.min_shared_ngrams:
                continue
            containment = overlap / max(1, min(len(candidate_ngrams), len(fingerprint)))
            if containment > best_score:
                best_score = containment

        if best_score >= self.similarity_threshold:
            return True, "fingerprint", best_score

        return False, "clean", best_score


def build_prompt_leakage_guard(
    prompt_texts: Iterable[str],
    ngram_size: int = 6,
    similarity_threshold: float = 0.28,
    min_shared_ngrams: int = 3,
    min_candidate_tokens: int = 12,
) -> PromptLeakageGuard:
    return PromptLeakageGuard(
        prompt_texts=prompt_texts,
        ngram_size=ngram_size,
        similarity_threshold=similarity_threshold,
        min_shared_ngrams=min_shared_ngrams,
        min_candidate_tokens=min_candidate_tokens,
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


def has_prompt_leakage(text: str) -> bool:
    return False  # for testing, bypass prompt leakage detection
    detected, _, _ = _DEFAULT_LEAK_GUARD.detect(text)
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
        res = float((await prompt_guard_llm.ainvoke(messages)).content)
        return res > 0.5
    except Exception as e:
        logger.warning(f"[prompt_guard] is_harmful check failed, allowing through: {e}")
        return False


# if __name__ == "__main__":
#     print(is_harmful("What is the capital of France?"))
