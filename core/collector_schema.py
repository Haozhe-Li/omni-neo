"""Request schema for the training-data collector (core/routers/collector.py).

The collector exists to produce training examples, and a training example is only
worth anything if the model saw *exactly* what production would have shown it.
So this module does not describe a friendlier input — it describes the same input
production produces, and rejects anything production could not have produced.

Each rule below is lifted from the client that talks to POST /chat today
(omni-neo-frontend: components/chat-view.tsx `buildPersonalization`,
lib/utils.ts `getLocalISOString`, lib/location.ts, components/settings-dialog.tsx):

- `user_local_datetime` is always present and is `getLocalISOString()`:
  `YYYY-MM-DDTHH:MM:SS±HH:MM`, local wall-clock time with its UTC offset.
- `user_location`, when present, is `"<city>, <country> (IP Approximate)"` or
  `"<city>, <country> (GPS Precise Location)"`. It is absent (not empty) when
  the browser could not locate itself.
- `response_language`, when present, is one of the settings dialog's raw codes.
  "Auto-detect" is stored as `auto` and is *omitted from the payload*, so the
  server default ("Follow User's Query Language") applies — `auto` is therefore
  not a valid value here either.
- `user_unit` is never sent, and `memory_enabled` is derived from the memory text
  rather than chosen, so neither is accepted.
- Nothing else a turn can carry (attachments, skills, source URLs, follow-up
  selections) is accepted: the collector UI does not expose them, and letting a
  field through before its production shape has been reviewed is how a dataset
  quietly stops matching production.

Pure on purpose (pydantic only), so tests/test_collector_parity.py can pin these
rules without the agent, the database or Redis.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.utils.data_model import Personalization, QueryRequest

# Models whose answers are teacher output. `best` serves luna today (see
# core/chat_models.py); `luna` is luna itself. Anything else is refused here
# rather than discovered at submit time, after a person has edited the answer.
COLLECTOR_MODELS: tuple[str, ...] = ("best", "luna")

# components/settings-dialog.tsx "Response language", minus `auto` (see above).
RESPONSE_LANGUAGES: tuple[str, ...] = ("en", "zh-CN", "zh-TW", "ja", "ko")

# lib/utils.ts getLocalISOString(): no fractional seconds, offset always numeric.
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
# lib/location.ts: `${city}, ${country} (IP Approximate | GPS Precise Location)`.
_LOCATION_RE = re.compile(r"^[^\n]+, [^\n]+ \((?:IP Approximate|GPS Precise Location)\)$")

MAX_QUERY_CHARS = 20_000
MAX_LOCATION_CHARS = 200


class CollectorPersonalization(BaseModel):
    """The slice of `Personalization` the production client sends, strictly."""

    model_config = ConfigDict(extra="forbid")

    user_local_datetime: str
    user_location: str | None = None
    response_language: Literal["en", "zh-CN", "zh-TW", "ja", "ko"] | None = None

    @field_validator("user_local_datetime")
    @classmethod
    def _datetime_is_production_format(cls, v: str) -> str:
        if not _DATETIME_RE.match(v):
            raise ValueError("must look like 2026-10-07T14:05:09+08:00 (lib/utils.ts getLocalISOString)")
        try:
            datetime.fromisoformat(v)
        except ValueError as e:
            raise ValueError(f"not a real date/time: {e}") from e
        return v

    @field_validator("user_location")
    @classmethod
    def _location_is_production_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if len(v) > MAX_LOCATION_CHARS or not _LOCATION_RE.match(v):
            raise ValueError(
                'must look like "Tokyo, Japan (IP Approximate)" or "... (GPS Precise Location)"; '
                "omit the field for a user whose location is unknown"
            )
        return v


class CollectorGenerateRequest(BaseModel):
    """One turn: the query plus the context production would attach to it."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    thread_id: str
    personalization: CollectorPersonalization
    model: str | None = None
    # Memory a human wrote for this conversation. First turn only: production
    # injects `<user_memory>` once and the checkpoint carries it from then on.
    memory: str | None = None

    @field_validator("query")
    @classmethod
    def _query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query is blank")
        return v

    @field_validator("model")
    @classmethod
    def _teacher_models_only(cls, v: str | None) -> str | None:
        if v is not None and v not in COLLECTOR_MODELS:
            raise ValueError(f"the collector only runs teacher models: {', '.join(COLLECTOR_MODELS)}")
        return v

    @field_validator("memory")
    @classmethod
    def _normalise_memory(cls, v: str | None) -> str | None:
        # save_user_memory stores `.strip()`; format_user_memory is skipped for "".
        v = (v or "").strip()
        return v or None


def to_query_request(req: CollectorGenerateRequest, *, turn: int) -> QueryRequest:
    """The `QueryRequest` the production client would have POSTed to /chat.

    Everything the generation path reads comes from this object, through the same
    functions /chat uses (core/utils/utils.py `build_turn_context`), so the two
    entry points cannot disagree about what a turn looks like.

    `memory_enabled` is true exactly when there is memory to inject — production's
    effective condition (enabled AND something stored).
    """
    p = req.personalization
    return QueryRequest(
        query=req.query,
        thread_id=req.thread_id,
        model=req.model,
        turn=turn,
        personalization=Personalization(
            memory_enabled=bool(req.memory),
            user_local_datetime=p.user_local_datetime,
            user_location=p.user_location,
            # `Personalization`'s own default when the client sends none.
            **({"response_language": p.response_language} if p.response_language else {}),
        ),
    )
