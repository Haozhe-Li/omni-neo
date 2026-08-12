"""The models a user can pick for a chat turn, and what each one costs.

This replaced the fast/pro *mode* switch. A mode was a bundle of prompt, turn
budget and skill roster; a model is just the weights. Everything else — the
system prompt, the 15 tools, all 9 skills, the 30-call budget — is now identical
across every entry here, which is what lets `rix` be served by a LoRA at
all (see `core/agent.py`'s Prompt sections block: an adapter has exactly one
compatible prompt).

Five entries, two of them open to guests:

    best      rix, auto-routed to gemma when the turn has an image
    rix  the fine-tune, text-only, no routing. Shown as "Rix" in the UI —
              the id is the wire value and is persisted, so it does not follow
              the display name.
    gemma     signed in
    luna      signed in
    gemini    signed in

## Billing

`rix` is 1 credit and everything else is 3, *including* a `best` turn that
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
    omni_pro_104_v1,
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
    # False only for `rix`: W&B serves the adapter text-only, so a
    # multimodal request 400s. The frontend blocks the attachment before it is
    # uploaded; `core/routers/chat.py` rejects it again for clients that don't.
    accepts_images: bool
    # Model to swap in when the conversation contains an image. Set on `best`
    # only — that swap *is* what "best available" means here. None elsewhere:
    # gemma/luna/gemini read images natively, and rix refuses them.
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
    "rix": ChatModel(
        id="rix",
        label="Rix",
        llm=omni_pro_104_v1,
        credits=1.0,
        requires_auth=False,
        accepts_images=False,
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

# Wire-level compatibility. Threads created before this change carry
# `mode: "fast" | "pro"`, and those values are persisted in message rows and in
# the frontend's localStorage — a rewind of an old thread will send one. Both
# map to `best`, which is the closest thing to what either used to do.
_LEGACY_ALIASES = {"fast": "best", "pro": "best"}


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
