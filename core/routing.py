"""Which concrete model serves a turn when the user picked `best`.

`best` is a router (core/chat_models.py). The policy, in priority order:

1. The user explicitly asked for a skill — web research, trip advisor or guided
   learning (`QueryRequest.skill`): luna, from this turn on.
2. The thread has already escalated: a skill was requested on an earlier turn
   (remembered in Redis, see `_escalated_from`): luna.
3. Exchange 6 and later: luna. The first five exchanges of a thread are rix.
4. Otherwise: rix.

"From this turn on" is sticky. Once a thread has gone to luna it stays there, so
a user who asks for web research and then follows up with a plain question is not
dropped back to rix mid-investigation.

The decision is made before anything is charged, because it *is* the billing key
(rix 0 credits, luna 3) — see core/chat_models.py. It needs the request plus one
Redis read.

Explicit picks (`rix`, `luna`, `gemini`) are not routed; they are served as asked.

## How the exchange number is known

The backend does not count turns; the frontend sends `turn`, the index its
assistant reply will land at in the message array (`[user, assistant, user, …]`,
so 1, 3, 5, … for exchanges 1, 2, 3, …). `exchange_number` converts. A client
that omits it or sends something small is routed as an early exchange — which is
free, and which any user can already get by picking Rix, so nothing here needs to
be tamper-proof.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from core.chat_models import CHAT_MODELS, ChatModel
from core.utils.redis_client import get_async_redis

logger = logging.getLogger(__name__)

EARLY_ROUTE = "rix"
LATE_ROUTE = "luna"

# The last exchange served by EARLY_ROUTE; exchange 6 is the first on LATE_ROUTE.
EARLY_EXCHANGES = 5

# Canonical skill names (after core.agent.resolve_skill_name) that force luna.
LUNA_SKILLS = frozenset({"web-research", "trip-advisor", "guided-learning"})

# Same lifetime as the thread-scoped skill markers in core/context_enrichment.py
# and as a signed-in user's thread retention.
_ESCALATED_TTL_S = 60 * 60 * 24 * 90


@dataclass(frozen=True)
class Route:
    model_id: str
    reason: str  # explicit | skill | escalated | late | early
    # Set when this turn is the one that escalates the thread; the caller stores
    # it with `remember_escalation` once the turn is actually going ahead.
    escalate_from: int | None = None


def exchange_number(turn: int | None) -> int | None:
    """1-indexed exchange for the frontend's `turn` (an array index: 1, 3, 5, …)."""
    if turn is None or turn < 1:
        return None
    return (turn + 1) // 2


def _key(thread_id: str) -> str:
    return f"route_escalated:{thread_id}"


async def _escalated_from(thread_id: str | None) -> int | None:
    """The exchange at which this thread first escalated, or None. Fails open:
    a Redis error means "not escalated", which routes to the free model."""
    if not thread_id:
        return None
    try:
        raw = await get_async_redis().get(_key(thread_id))
        return int(raw) if raw is not None else None
    except Exception as e:
        logger.warning(f"[routing] escalation read failed for {thread_id}: {e}")
        return None


def pick(exchange: int | None, skill: str | None, escalated_from: int | None) -> tuple[str, str]:
    """The pure policy: (model id, reason)."""
    if skill in LUNA_SKILLS:
        return LATE_ROUTE, "skill"
    if escalated_from is not None and (exchange is None or exchange >= escalated_from):
        return LATE_ROUTE, "escalated"
    if exchange is not None and exchange > EARLY_EXCHANGES:
        return LATE_ROUTE, "late"
    return EARLY_ROUTE, "early"


async def decide_route(
    model: ChatModel, thread_id: str | None, turn: int | None, skill: str | None
) -> Route:
    """The concrete model for this turn. `skill` is the canonical skill name."""
    if not model.is_router:
        return Route(model.id, "explicit")
    exchange = exchange_number(turn)
    escalated_from = await _escalated_from(thread_id)
    model_id, reason = pick(exchange, skill, escalated_from)
    escalate_from = None
    if reason == "skill" and thread_id and (escalated_from is None or (exchange or 1) < escalated_from):
        escalate_from = exchange or 1
    return Route(model_id, reason, escalate_from)


async def remember_escalation(thread_id: str | None, route: Route) -> None:
    """Persist that the thread escalated, so later turns stay on luna even
    without the skill. Call once the turn is committed, not before the gates."""
    if not thread_id or route.escalate_from is None:
        return
    try:
        await get_async_redis().set(_key(thread_id), route.escalate_from, ex=_ESCALATED_TTL_S)
    except Exception as e:
        logger.warning(f"[routing] escalation write failed for {thread_id}: {e}")


def serving(route: Route) -> ChatModel:
    return CHAT_MODELS[route.model_id]
