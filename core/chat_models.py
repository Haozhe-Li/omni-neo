"""The models a user can pick for a chat turn, and what each one costs.

This replaced the fast/pro *mode* switch. A mode was a bundle of prompt, turn
budget and skill roster; a model is just the weights. Everything else — the
system prompt, the tools, all 9 skills, the tool budget — is identical
across every entry here. That uniformity is what lets `rix` be served by a LoRA
(an adapter has exactly one compatible prompt), and keeping the entries uniform
is what makes serving the next one a one-line change.

Four entries, two of them open to guests:

    best      the default; luna underneath, on both the text and image paths
    rix       the fine-tune: Gemma 4 26B-A4B distilled from luna (opt-in)
    luna      signed in
    gemini    signed in

## Billing

The user pays for the model that actually runs, never for the name they picked.
`best` is a router, not a model: it has no price of its own, and a turn on it is
billed as whichever concrete model `serving_model` says takes the turn — luna is
3 credits, rix is 1. Today `best` routes to luna on both the text and image
paths, so a `best` turn costs 3. When the fine-tune takes the text path again,
pointing `text_route` back at `rix` is the whole change: the price follows.

The routing decision and the billing decision are made in different places, and
that is the seam worth knowing about. `VisionModelMiddleware` swaps when an
image appears **anywhere in the conversation**; billing runs before the agent
does and can only see **this turn's** attachments. So once `best`'s two routes
differ in price, a follow-up question about an image sent two turns ago would be
billed on the text route. Closing that gap means reading thread state on the
charge path, which is a DB round trip on the hot path for a rare case —
`serving_model` documents the rule it actually implements rather than pretending
otherwise. (While both routes are luna the gap costs nothing.)
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from core.llm import (
    gemini_3_6_flash,
    gpt_6_luna,
    rix_gemma_v1,
    vision_llm,
)


@dataclass(frozen=True)
class ChatModel:
    id: str
    label: str
    llm: BaseChatModel
    # Per-turn price of a concrete model. None on a router (`best`), which is
    # billed as whatever it routes to — see `serving_model`.
    credits: float | None
    requires_auth: bool
    # True for every model currently listed. It exists for text-only entries:
    # the frontend blocks the attachment before it is uploaded, and
    # `core/routers/chat.py` rejects it again for clients that don't.
    accepts_images: bool
    # Model to swap in when the conversation contains an image. Set on `best`
    # only — that swap *is* what "best available" means here. None elsewhere:
    # luna/gemini/rix read images natively.
    vision_fallback: BaseChatModel | None = None
    # Routers only: the id of the concrete CHAT_MODELS entry that takes a text
    # turn / an image turn. These decide the price. They must name the same
    # model `llm` / `vision_fallback` actually are, or the user is billed for one
    # model and served another.
    text_route: str | None = None
    vision_route: str | None = None


CHAT_MODELS: dict[str, ChatModel] = {
    "best": ChatModel(
        id="best",
        label="Best",
        # Luna on both paths for now. The fine-tune served the text half until
        # it started misbehaving in production; `best` is the default and the
        # only entry a guest can reach, so it is the one place that cannot
        # carry a model under investigation. Picking `rix` explicitly still
        # gets the adapter, and so does the eval matrix — `core/llm.py`'s
        # `chat_llm` deliberately did not follow this switch.
        llm=gpt_6_luna,
        # A router has no price of its own; see `text_route` / `vision_route`.
        credits=None,
        requires_auth=False,
        accepts_images=True,
        # A swap to the model already running, and kept anyway: it is the
        # line that has to change back when the adapter returns.
        vision_fallback=vision_llm,
        text_route="luna",
        vision_route="luna",
    ),
    "rix": ChatModel(
        id="rix",
        label="Rix",
        llm=rix_gemma_v1,
        credits=1.0,
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


def serving_model(model_id: str | None, *, has_image: bool = False) -> ChatModel:
    """The concrete model that will run this turn — what the user is billed for.

    A concrete model serves itself. A router (`best`) resolves through its
    `text_route` / `vision_route`. `has_image` means *this request* carries an
    image attachment — the only signal available before the agent runs. See the
    module docstring for the follow-up-turn gap this leaves open.
    """
    m = resolve_model(model_id)
    if m.text_route is None:
        return m
    route = m.vision_route if has_image and m.vision_route else m.text_route
    return CHAT_MODELS[route]


def credits_for(model_id: str | None, *, has_image: bool = False) -> float:
    """Credit cost of one turn on `model_id`, after routing."""
    return serving_model(model_id, has_image=has_image).credits
