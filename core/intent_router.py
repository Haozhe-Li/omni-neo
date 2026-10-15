"""Intent router — one multiple-choice question put to OpenAI's Decisions API.

Replaces the embedding classifier (anchors + a softmax head + a Redis cache). The
user's message goes to `POST /v1/decisions` with a single `choice` question whose
options are core/intent_choices.py; the API always answers with one of them and a
probability for each. There is no "unsure" outcome and so no model to fall back
to on a low score: the caller acts on the label it gets.

Two things are decided here, both from the returned probabilities:

- The label is the API's top choice.
- ...except a skill label (`skill:<name>`), which needs `MIN_PROB_SKILL`. A false
  positive loads a ~12k-char workflow and sends the agent off on a research run for
  what was a one-shot question, while a miss costs nothing (the agent can still load
  the skill itself). Below the bar the best *non-skill* option wins — still the API's
  own answer, not a second model.

What can still fail is the request itself (timeout, network, a refusal, an API
change, no beta access). That raises `RouterError` and the caller treats it as "the
router is down" and lets the scout LLM classify the turn. The call is on the critical
path of every turn, so the caller owns the timeout.

The API is in public beta and only serves `gpt-6-luna`. Plain httpx rather than the
OpenAI SDK: the SDK version that has this endpoint is a major release above the one
`langchain-openai` is pinned against.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import httpx

from core.intent_choices import INSTRUCTIONS, INTENT_CHOICES, SKILL_LABEL_PREFIX

logger = logging.getLogger(__name__)

MODEL = "gpt-6-luna"
_URL = os.getenv("INTENT_ROUTER_URL", "https://api.openai.com/v1/decisions")
_QUESTION = "intent"

# Skill labels get the highest bar of all: see the module docstring. Not calibrated
# against a large set (there are few skill cases in evals/scout_routing) — re-run
# `python evals/scout_routing/run.py --router` before lowering it.
MIN_PROB_SKILL = float(os.getenv("INTENT_ROUTER_MIN_PROB_SKILL", "0.8"))

_CHOICES = [{"value": v, "description": d} for v, d in INTENT_CHOICES.items()]


class RouterError(RuntimeError):
    """The Decisions request did not produce a usable answer."""


@dataclass(frozen=True)
class RouteResult:
    """The outcome of routing one query. ``label`` is always one of the options;
    ``score`` is its probability and ``scores`` the probability of every option."""

    label: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        # The caller's wait_for is the real deadline; this only bounds a hung socket.
        _client = httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=2.0))
    return _client


def pick(scores: dict[str, float]) -> str:
    """The label for a set of option probabilities (see the module docstring)."""
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    top = ranked[0]
    if top.startswith(SKILL_LABEL_PREFIX) and scores[top] < MIN_PROB_SKILL:
        return next(lab for lab in ranked if not lab.startswith(SKILL_LABEL_PREFIX))
    return top


def parse(payload: dict) -> RouteResult:
    """A `RouteResult` from a Decisions response body; `RouterError` if it has no
    usable answer to our question."""
    answers = payload.get("answers") if isinstance(payload, dict) else None
    ans = next((a for a in answers or [] if isinstance(a, dict) and a.get("name") == _QUESTION), None)
    if ans is None or ans.get("type") != "choice":
        raise RouterError(f"no choice answer in response (got {ans and ans.get('type')!r})")
    scores = {
        p["value"]: float(p["probability"])
        for p in ans.get("probabilities") or []
        if p.get("value") in INTENT_CHOICES
    }
    if not scores:
        # No probabilities: trust the choice itself, if it is one of ours.
        if ans.get("choice") not in INTENT_CHOICES:
            raise RouterError(f"unknown choice {ans.get('choice')!r}")
        label = ans["choice"]
        return RouteResult(label=label, score=float(ans.get("confidence") or 1.0), scores={label: 1.0})
    label = pick(scores)
    return RouteResult(label=label, score=scores[label], scores=scores)


async def route(query: str, *, client: httpx.AsyncClient | None = None) -> RouteResult:
    """Classify one query. Raises `RouterError` (or an httpx error) if the request fails."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RouterError("OPENAI_API_KEY is not set")
    resp = await (client or _get_client()).post(
        _URL,
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": MODEL,
            "input": query,
            "questions": [{
                "type": "choice",
                "name": _QUESTION,
                "instructions": INSTRUCTIONS,
                "choices": _CHOICES,
            }],
        },
    )
    if resp.status_code != 200:
        raise RouterError(f"decisions API returned {resp.status_code}: {resp.text[:200]}")
    return parse(resp.json())
