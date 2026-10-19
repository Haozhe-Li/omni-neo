"""The models a user can pick for a chat turn, and what each one costs.

This replaced the fast/pro *mode* switch. A mode was a bundle of prompt, turn
budget and skill roster; a model is just the weights. Everything else — the
system prompt, the tools, all 9 skills, the tool budget — is identical
across every entry here. That uniformity is what lets `rix` be served by a LoRA
(an adapter has exactly one compatible prompt), and keeping the entries uniform
is what makes serving the next one a one-line change.

Four entries, three of them open to guests:

    best      the default; a router — see core/routing.py. It has no weights of
              its own: each turn lands on rix or luna
    rix       the fine-tune: Gemma 4 26B-A4B distilled from luna (free)
    luna      signed in
    gemini    signed in

## Billing

The user pays for the model that actually runs, never for the name they picked:
rix is free (0 credits), luna and gemini are 3. `best` has no price of its own —
core/routing.py decides which concrete model takes the turn *before* anything is
charged (routing needs only the request and one Redis read), and that model's id
is the billing key. A turn that lands on rix costs nothing, so it also passes the
usage gate for a user who is out of credits.

`serving_model` is the answer for a caller that does not run the routing policy
(the training-data collector, which wants the teacher; a state read that needs
*an* agent): a router resolves to its `fallback_route`, luna.

Images do not change the route or the price. Every model reads them, rix
included (Gemma 4 is multimodal and W&B serves the adapter with vision).
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from core.llm import (
    gemini_3_6_flash,
    gpt_6_luna,
    rix_gemma_v1,
)


@dataclass(frozen=True)
class ChatModel:
    id: str
    label: str
    # None on a router (`best`): it has no weights of its own, so no agent is
    # built for it — turns run on the concrete model core/routing.py picks.
    llm: BaseChatModel | None
    # Per-turn price of a concrete model; 0 is free. None on a router, which is
    # billed as whatever it routes to.
    credits: float | None
    requires_auth: bool
    # True for every model currently listed. It exists for text-only entries:
    # the frontend blocks the attachment before it is uploaded, and
    # `core/routers/chat.py` rejects it again for clients that don't.
    accepts_images: bool
    # Routers only: the concrete model for callers that do not run the routing
    # policy (see `serving_model`).
    fallback_route: str | None = None

    @property
    def is_router(self) -> bool:
        return self.fallback_route is not None


CHAT_MODELS: dict[str, ChatModel] = {
    "best": ChatModel(
        id="best",
        label="Best",
        # A router: core/routing.py sends the early turns of a thread to rix and
        # the rest (and anything that asked for a skill) to luna. No llm and no
        # price of its own.
        llm=None,
        credits=None,
        requires_auth=False,
        accepts_images=True,
        fallback_route="luna",
    ),
    "rix": ChatModel(
        id="rix",
        label="Rix",
        llm=rix_gemma_v1,
        credits=0.0,
        requires_auth=False,
        # Gemma 4 is multimodal and W&B serves this adapter with vision (an image
        # turn answered correctly when checked), and the training set includes
        # image turns, so no swap to another model is needed.
        accepts_images=True,
    ),
    "luna": ChatModel(
        id="luna",
        label="GPT-6 Luna",
        llm=gpt_6_luna,
        credits=3.0,
        requires_auth=True,
        accepts_images=True,
    ),
    "gemini": ChatModel(
        id="gemini",
        label="Gemini 3.6 Flash",
        llm=gemini_3_6_flash,
        credits=3.0,
        requires_auth=True,
        accepts_images=True,
    ),
}

DEFAULT_MODEL = "best"

# Wire-level compatibility. Persisted message rows and the frontend's
# localStorage still carry `mode: "fast" | "pro"` from before the mode/model
# switch; a rewind of an old thread will send one. Both map to `best`, the
# closest thing to what each used to do.
#
# `rix` is deliberately NOT here any more. It was aliased to `best` while the
# fine-tune was offline, and this lookup runs *before* the CHAT_MODELS one — so
# leaving it would silently shadow the real `rix` entry above and serve `best`
# to everyone who picked the adapter.
#
# `gemma` is here because Cerebras archived gemma-4-31b and the entry was
# removed, but threads created while it was selectable still carry it, and
# `resolve_model` raises on an unknown id — a rewind of one would 400.
_LEGACY_ALIASES = {"fast": "best", "pro": "best", "gemma": "best"}


def resolve_model(model_id: str | None) -> ChatModel:
    """Map a client-supplied id onto a model. Unknown ids raise ValueError.

    Deliberately strict rather than falling back to the default: a typo that
    silently downgrades a signed-in user to `best` is invisible to them and
    bills differently than what they asked for.
    """
    key = (model_id or DEFAULT_MODEL).strip().lower()
    key = _LEGACY_ALIASES.get(key, key)
    if key not in CHAT_MODELS:
        raise ValueError(
            f"unknown model {model_id!r} — expected one of {sorted(CHAT_MODELS)}"
        )
    return CHAT_MODELS[key]


def serving_model(model_id: str | None) -> ChatModel:
    """The concrete model behind `model_id` for a caller that does not run the
    routing policy: a concrete model is itself, a router is its fallback."""
    m = resolve_model(model_id)
    return CHAT_MODELS[m.fallback_route] if m.is_router else m


def credits_for(model_id: str | None) -> float:
    """Credit cost of one turn on `model_id` (a router priced as its fallback)."""
    return serving_model(model_id).credits
