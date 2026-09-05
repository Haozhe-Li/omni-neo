"""The models a user can pick for a chat turn, and what each one costs.

This replaced the fast/pro *mode* switch. A mode was a bundle of prompt, turn
budget and skill roster; a model is just the weights. Everything else — the
system prompt, the tools, all 9 skills, the 30-call budget — is identical
across every entry here. That uniformity was originally what let `rix` be
served by a LoRA (an adapter has exactly one compatible prompt); `rix` is
offline pending a retrain against the tool adapter layer, but keeping the
entries uniform is what makes serving the next one a one-line change.

Four entries, one of them open to guests:

    best      the default, auto-routed to gemma when the turn has an image
    gemma     signed in
    luna      signed in
    gemini    signed in

## Billing

`best` is 1 credit and everything else is 3, *including* a `best` turn that
routes to gemma — the user pays for the model that actually ran, not the one
they picked.

The routing decision and the billing decision are made in different places, and
that is the one seam worth knowing about. `VisionModelMiddleware` swaps to gemma
when an image appears **anywhere in the conversation**; billing runs before the
agent does and can only see **this turn's** attachments. So a follow-up question
about an image sent two turns ago is served by gemma and billed at 1 credit.
Closing that gap means reading thread state on the charge path, which is a DB
round trip on the hot path for a rare case — `credits_for` documents the rule it
actually implements rather than pretending otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from core.llm import (
    chat_llm,
    gemini_3_6_flash,
    gemma_4_31b,
    gpt_5_6_luna,
    vision_llm,
)


@dataclass(frozen=True)
class ChatModel:
    id: str
    label: str
    llm: BaseChatModel
    credits: float
    requires_auth: bool
    # True for every model currently listed. It exists for text-only entries
    # like the offline `rix` fine-tune, which W&B serves without vision: the
    # frontend blocks the attachment before it is uploaded, and
    # `core/routers/chat.py` rejects it again for clients that don't.
    accepts_images: bool
    # Model to swap in when the conversation contains an image. Set on `best`
    # only — that swap *is* what "best available" means here. None elsewhere:
    # gemma/luna/gemini read images natively.
    vision_fallback: BaseChatModel | None = None
    # Credits charged when `vision_fallback` takes the turn.
    vision_credits: float | None = None


CHAT_MODELS: dict[str, ChatModel] = {
    "best": ChatModel(
        id="best",
        label="Best",
        llm=chat_llm,
        credits=1.0,
        requires_auth=False,
        accepts_images=True,
        vision_fallback=vision_llm,
        vision_credits=3.0,
    ),
    "gemma": ChatModel(
        id="gemma",
        label="Gemma 4",
        llm=gemma_4_31b,
        credits=3.0,
        requires_auth=True,
        accepts_images=True,
    ),
    "luna": ChatModel(
        id="luna",
        label="GPT-5.6 Luna",
        llm=gpt_5_6_luna,
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
# localStorage still carry ids this table no longer lists: `mode: "fast" |
# "pro"` from before the mode/model switch, and `rix` from before the
# fine-tune was taken offline. A rewind of an old thread will send one. All
# map to `best`, the closest thing to what each used to do.
_LEGACY_ALIASES = {"fast": "best", "pro": "best", "rix": "best"}


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


def credits_for(model_id: str | None, *, has_image: bool = False) -> float:
    """Credit cost of one turn on `model_id`.

    `has_image` means *this request* carries an image attachment — the only
    signal available before the agent runs. See the module docstring for the
    follow-up-turn gap this leaves open.
    """
    m = resolve_model(model_id)
    if has_image and m.vision_credits is not None:
        return m.vision_credits
    return m.credits
