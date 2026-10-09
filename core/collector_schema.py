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
- `skill`, when present, is one of the three ids the chat's skill picker offers
  (components/chat-view.tsx `SKILLS`); the backend maps it to the skill on disk
  with the same `resolve_skill_name` /chat uses. The other skills in skills/ are
  chosen by the agent or the scout, never by the user, so they are not accepted.
- `attached_file_ids` is `[{<file_id>: <filename>}, ...]`, one dict per file, ready files
  only, at most 5 (search-home.tsx / chat-view.tsx); the files themselves go through
  the same upload minting `/api/upload` uses and the same allow-list and 20 MB cap the
  composers enforce (lib/upload-types.ts). A turn may have no text if it has files
  (chat-view's send), and the router checks every id belongs to this thread.
- `source_url` is the "Add URL" list: at most 5, each an http(s) URL in the canonical
  form `new URL(...).toString()` leaves (hooks/useSourceUrls.ts `normalizeUrl`).
- Nothing else a turn can carry (follow-up selections) is accepted: the collector UI does not expose them, and letting a
  field through before its production shape has been reviewed is how a dataset
  quietly stops matching production.

Pure on purpose (pydantic only), so tests/test_collector_parity.py can pin these
rules without the agent, the database or Redis.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

# components/chat-view.tsx `SKILLS`: the only skills a user can switch on. The wire
# value is the id; the backend aliases `deep-research` to the `web-research` skill.
SKILL_IDS: tuple[str, ...] = ("deep-research", "trip-advisor", "guided-learning")

# ── uploads: mirrors omni-neo-frontend lib/upload-types.ts and search-home.tsx ──
# What a composer lets through. Extension OR MIME, exactly like `isAllowedUploadFile`,
# because browsers report generic MIME types for many office formats.
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_FILES_PER_TURN = 5
MAX_SOURCE_URLS = 5

_DOCUMENT_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".docm",
    ".ppt", ".pps", ".pot", ".pptx", ".pptm", ".ppsx", ".ppsm",
    ".xls", ".xlsx", ".xlsm", ".xlsb", ".odt", ".ods", ".odp", ".rtf", ".epub",
}
_DOCUMENT_MIME_TYPES = {
    "application/pdf", "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-word.document.macroEnabled.12", "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.ms-powerpoint.presentation.macroEnabled.12",
    "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
    "application/vnd.ms-powerpoint.slideshow.macroEnabled.12", "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel.sheet.macroEnabled.12",
    "application/vnd.ms-excel.sheet.binary.macroEnabled.12",
    "application/vnd.oasis.opendocument.text", "application/vnd.oasis.opendocument.spreadsheet",
    "application/vnd.oasis.opendocument.presentation", "application/rtf", "text/rtf",
    "application/epub+zip",
}
_TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".py", ".js", ".jsx", ".ts", ".tsx", ".html",
    ".json", ".xml", ".yaml", ".yml", ".java", ".c", ".cpp", ".h", ".hpp", ".sh",
}
_TEXT_MIME_TYPES = {
    "text/plain", "text/markdown", "text/html", "application/json",
    "application/xml", "text/xml", "application/yaml", "application/x-yaml", "text/yaml",
}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}

_ALLOWED_EXTENSIONS = _DOCUMENT_EXTENSIONS | _TEXT_EXTENSIONS | _IMAGE_EXTENSIONS
_ALLOWED_MIME_TYPES = _DOCUMENT_MIME_TYPES | _TEXT_MIME_TYPES | _IMAGE_MIME_TYPES


def is_allowed_upload(filename: str, file_type: str) -> bool:
    """`isAllowedUploadFile` of lib/upload-types.ts."""
    ext = filename[filename.rfind("."):].lower() if "." in filename else ""
    return file_type in _ALLOWED_MIME_TYPES or ext in _ALLOWED_EXTENSIONS


# `user_uploads/<owner>/<uuid>`, as core/routers/uploads.py mints it.
_FILE_ID_RE = re.compile(r"^user_uploads/[a-z0-9_-]{1,40}/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def _is_canonical_url(url: str) -> bool:
    """A URL as `new URL(raw).toString()` writes it: http(s), a host, lower-case
    scheme and host, and at least a `/` path — what `normalizeUrl` hands the API."""
    if len(url) > 2048 or re.search(r"\s", url):
        return False
    # urlparse lower-cases the scheme for us, so test the text itself.
    if not url.startswith(("http://", "https://")):
        return False
    p = urlparse(url)
    if not p.hostname or not p.path:
        return False
    host = p.netloc.rsplit("@", 1)[-1].rsplit(":", 1)[0] if not p.netloc.endswith("]") else p.netloc
    return host == host.lower()


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


class CollectorUploadRequest(BaseModel):
    """`POST /api/upload/url`'s body, held to what the composers let a user attach."""

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    file_type: str = Field(max_length=200)
    file_size_bytes: int = Field(ge=1, le=MAX_UPLOAD_BYTES)

    @model_validator(mode="after")
    def _allowed(self):
        if "/" in self.filename or "\\" in self.filename or "\x00" in self.filename:
            raise ValueError("filename must not contain a path")
        if not is_allowed_upload(self.filename, self.file_type):
            raise ValueError("not a file type the composers accept")
        return self


class CollectorGenerateRequest(BaseModel):
    """One turn: the query plus the context production would attach to it."""

    model_config = ConfigDict(extra="forbid")

    # May be empty when files are attached (chat-view's send allows it); see below.
    query: str = Field(max_length=MAX_QUERY_CHARS)
    thread_id: str
    personalization: CollectorPersonalization
    model: str | None = None
    # The skill the user switched on for this turn, as the picker sends it. The
    # picker keeps a skill on until it is cleared, so a client carrying it across
    # turns is the production shape too.
    skill: Literal["deep-research", "trip-advisor", "guided-learning"] | None = None
    # `[{file_id: filename}]`, one dict per file — the shape chat-view sends.
    attached_file_ids: list[dict[str, str]] | None = Field(default=None, max_length=MAX_FILES_PER_TURN)
    # The "Add URL" list, already normalised by the client.
    source_url: list[str] | None = Field(default=None, max_length=MAX_SOURCE_URLS)
    # Memory a human wrote for this conversation. First turn only: production
    # injects `<user_memory>` once and the checkpoint carries it from then on.
    memory: str | None = None

    @field_validator("attached_file_ids")
    @classmethod
    def _files_are_the_production_shape(cls, v):
        if not v:
            return None
        seen: set[str] = set()
        for entry in v:
            if len(entry) != 1:
                raise ValueError("each attached_file_ids entry is one {file_id: filename}")
            ((file_id, name),) = entry.items()
            if not _FILE_ID_RE.match(file_id) or not (1 <= len(name) <= 255):
                raise ValueError("not a file id / filename the upload endpoint mints")
            if file_id in seen:
                raise ValueError("a file is attached twice")
            seen.add(file_id)
        return v

    @field_validator("source_url")
    @classmethod
    def _urls_are_canonical(cls, v):
        if not v:
            return None
        for url in v:
            if not _is_canonical_url(url):
                raise ValueError(f"not a normalised http(s) URL: {url[:80]!r}")
        if len(set(v)) != len(v):
            raise ValueError("a URL is listed twice")
        return v

    @model_validator(mode="after")
    def _something_to_say(self):
        # chat-view lets a turn go out with no text only when files ride along; a
        # URL-only turn is not sendable there, and the first turn gets its text from
        # the client ("Please read this file").
        if not self.query.strip() and not self.attached_file_ids:
            raise ValueError("query is blank")
        return self

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
        skill=req.skill,
        attached_file_ids=req.attached_file_ids,
        source_url=req.source_url,
        turn=turn,
        personalization=Personalization(
            memory_enabled=bool(req.memory),
            user_local_datetime=p.user_local_datetime,
            user_location=p.user_location,
            # `Personalization`'s own default when the client sends none.
            **({"response_language": p.response_language} if p.response_language else {}),
        ),
    )
