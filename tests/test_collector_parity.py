"""The collector must feed the agent exactly what production feeds it.

    venv/bin/python3.12 -m pytest tests/test_collector_parity.py -q

A collected example is only training data if the model saw, in the user message,
what it would see for a real user. These tests pin that from three sides:

1. the collector *rejects* anything the production client could not have sent
   (formats lifted from omni-neo-frontend: lib/utils.ts getLocalISOString,
   lib/location.ts, components/settings-dialog.tsx, chat-view.tsx
   buildPersonalization);
2. for inputs it accepts, the system reminder and memory block are byte-identical
   to what POST /chat builds from the payload the frontend would have sent;
3. the memory rule (first turn only) is the same function in both entry points.

The last class goes one step further and renders the whole user message with
`build_message_content` — the function that lays the blocks out — so a change to
the prompt layout shows up as a diff against a literal here, not as a silent
shift in the dataset.
"""

from __future__ import annotations

import os

import pytest
from pydantic import ValidationError

from core.collector_schema import (
    COLLECTOR_MODELS,
    CollectorGenerateRequest,
    CollectorUploadRequest,
    to_query_request,
)
from core.utils.data_model import QueryRequest
from core.utils.utils import build_turn_context, memory_injection_due

# Importing core.stream builds every LLM/search/storage client at import time. None
# of them is called here (the layout under test is a pure function), but the
# constructors want *something*; real values from the environment win.
_IMPORT_ENV = {
    "REDIS_URL": "redis://127.0.0.1:6399/0",
    "DATABASE_URL": "postgresql://x:x@127.0.0.1:1/x",
    "S3_ENDPOINT_URL": "http://127.0.0.1:9",
    "QDRANT_URL": "http://127.0.0.1:6333",
    "EMBEDDING_SERVICE_URL": "http://127.0.0.1:9",
    **{k: "test" for k in (
        "TAVILY_API_KEY", "WANDB_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
        "CEREBRAS_API_KEY", "EXA_API_KEY", "QSTASH_TOKEN", "OPENWEATHERMAP_API_KEY",
        "E2B_API_KEY", "FISH_API_KEY", "RESEND_API_KEY", "SPIDER_API_KEY", "QDRANT_API_KEY",
        "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_BUCKET_NAME",
    )},
}

FID = "user_uploads/collector_alice/0b8f6a52-3c41-4a77-9d0e-6f5d1c2e9a10"
FID2 = "user_uploads/collector_alice/9a1c2d3e-4b5f-4a6b-8c7d-0e1f2a3b4c5d"

DT = "2026-10-07T14:05:09+08:00"
LOC = "Shanghai, China (IP Approximate)"
MEMORY = "## Profile\n- Backend engineer, prefers terse answers\n\n## Current Focus\n- Migrating to Postgres 17"


def collector_request(**over) -> CollectorGenerateRequest:
    base = dict(
        query="今天适合跑步吗？",
        thread_id="t-1",
        personalization={"user_local_datetime": DT, "user_location": LOC, "response_language": "zh-CN"},
        model="best",
        memory=MEMORY,
    )
    base.update(over)
    return CollectorGenerateRequest(**base)


def frontend_payload(*, turn: int, with_memory: bool = True, lang="zh-CN", loc=LOC) -> dict:
    """What components/chat-view.tsx `runQuery` POSTs to /chat, field for field.

    `buildPersonalization` omits response_language for 'auto', sets memory_enabled
    only when the user has memory switched on, always sets the datetime, and sets
    the location only when it could be resolved. `payload.model` is the picker's
    id and `turn` is `baseHistory.length`.
    """
    p: dict = {"user_local_datetime": DT}
    if lang:
        p["response_language"] = lang
    if with_memory:
        p["memory_enabled"] = True
    if loc:
        p["user_location"] = loc
    return {"query": "今天适合跑步吗？", "thread_id": "t-1", "model": "best", "turn": turn, "personalization": p}


# ── 1. nothing production could not send gets through ──────────────────────


class TestRejectsNonProduction:
    def test_accepts_the_production_shape(self):
        collector_request()
        collector_request(personalization={"user_local_datetime": DT})  # no location, no language
        collector_request(memory=None)

    @pytest.mark.parametrize(
        "dt",
        [
            "2026-10-07T14:05:09Z",               # production always sends a numeric offset
            "2026-10-07T14:05:09.123+08:00",      # getLocalISOString has no fractions
            "2026-10-07 14:05:09+08:00",          # 'T' separator
            "2026-10-07T14:05+08:00",             # seconds are always present
            "2026-13-07T14:05:09+08:00",          # not a real date
            "",
        ],
    )
    def test_datetime_must_be_getLocalISOString(self, dt):
        with pytest.raises(ValidationError):
            collector_request(personalization={"user_local_datetime": dt})

    @pytest.mark.parametrize(
        "loc",
        ["Shanghai", "Shanghai, China", "Shanghai, China (approximate)", "Shanghai (IP Approximate)", "\n, x (IP Approximate)"],
    )
    def test_location_must_carry_the_source_suffix(self, loc):
        with pytest.raises(ValidationError):
            collector_request(personalization={"user_local_datetime": DT, "user_location": loc})

    @pytest.mark.parametrize("loc", ["Tokyo, Japan (GPS Precise Location)", "Unknown City, Unknown Country (IP Approximate)"])
    def test_location_accepts_both_production_suffixes(self, loc):
        collector_request(personalization={"user_local_datetime": DT, "user_location": loc})

    @pytest.mark.parametrize("lang", ["auto", "fr", "zh", "English", ""])
    def test_language_must_be_a_settings_code(self, lang):
        # 'auto' is stored by the settings dialog but *omitted* from the payload
        with pytest.raises(ValidationError):
            collector_request(personalization={"user_local_datetime": DT, "response_language": lang})

    @pytest.mark.parametrize("field", ["user_unit", "memory_enabled", "skill", "tz"])
    def test_unknown_personalization_fields(self, field):
        with pytest.raises(ValidationError):
            collector_request(personalization={"user_local_datetime": DT, field: "x"})

    @pytest.mark.parametrize("field", ["follow_up_content", "turn", "mode"])
    def test_unknown_request_fields(self, field):
        # Each is something a turn can carry in production that the collector does not
        # model; letting one through unreviewed is how a dataset drifts.
        with pytest.raises(ValidationError):
            CollectorGenerateRequest(
                query="q", thread_id="t", personalization={"user_local_datetime": DT}, **{field: "x"}
            )

    @pytest.mark.parametrize("skill", ["deep-research", "trip-advisor", "guided-learning", None])
    def test_skill_accepts_the_pickers_ids(self, skill):
        assert collector_request(skill=skill).skill == skill

    @pytest.mark.parametrize(
        "skill",
        # the on-disk name (the picker never sends it), skills only the agent picks, near-misses
        ["web-research", "charting", "mapping", "ask-question", "Deep Research", "deep_research", "", "none"],
    )
    def test_skill_rejects_anything_the_picker_cannot_send(self, skill):
        with pytest.raises(ValidationError):
            collector_request(skill=skill)

    # ── files ──
    def test_attached_files_in_the_chat_view_shape(self):
        r = collector_request(attached_file_ids=[{FID: "report.pdf"}, {FID2: "chart.png"}])
        assert r.attached_file_ids == [{FID: "report.pdf"}, {FID2: "chart.png"}]

    @pytest.mark.parametrize(
        "bad",
        [
            [{FID: "a.pdf", FID2: "b.pdf"}],                     # one dict per file
            [{"../etc/passwd": "a.pdf"}],                        # not an id the upload endpoint mints
            [{"user_uploads/x/not-a-uuid": "a.pdf"}],
            [{"other_prefix/collector_alice/0b8f6a52-3c41-4a77-9d0e-6f5d1c2e9a10": "a.pdf"}],
            [{FID: "a.pdf"}, {FID: "a.pdf"}],                    # twice
            [{FID: ""}],                                         # no name
            [{FID: "x" * 256}],
            [{FID: "a.pdf"}] * 6,                                # over the 5-file cap
            [],                                                  # production omits the field instead
        ],
    )
    def test_attached_files_the_composer_could_not_send(self, bad):
        # `[]` normalises to "no files" (production omits the key) rather than failing
        if bad == []:
            assert collector_request(attached_file_ids=bad).attached_file_ids is None
            return
        with pytest.raises(ValidationError):
            collector_request(attached_file_ids=bad)

    def test_a_turn_may_have_no_text_only_with_files(self):
        assert collector_request(query="", attached_file_ids=[{FID: "a.pdf"}]).query == ""
        for q in ("", "  \n"):
            with pytest.raises(ValidationError):
                collector_request(query=q)
            with pytest.raises(ValidationError):
                collector_request(query=q, source_url=["https://example.com/"])   # URL-only is not sendable

    # ── URLs ──
    @pytest.mark.parametrize("url", ["https://example.com/", "http://example.com/a?b=1#c", "https://en.wikipedia.org/wiki/Rapid_transit",
                                     "https://omniknows.xyz/pages/abc", "https://user@host.test:8080/p"])
    def test_source_urls_in_normalised_form(self, url):
        assert collector_request(source_url=[url]).source_url == [url]

    @pytest.mark.parametrize(
        "bad",
        [
            "example.com", "www.example.com/a",                  # normalizeUrl adds the scheme before sending
            "ftp://example.com/", "javascript:alert(1)", "file:///etc/passwd",
            "https://example.com",                               # new URL(...).toString() writes a "/"
            "HTTPS://example.com/", "https://EXAMPLE.com/",      # ... and lower-cases scheme and host
            "https://exa mple.com/", "https://example.com/a b",  # no whitespace
            "https://" + "a" * 2050 + ".com/",
            "",
        ],
    )
    def test_source_urls_the_picker_could_not_send(self, bad):
        with pytest.raises(ValidationError):
            collector_request(source_url=[bad])

    def test_source_url_caps_and_duplicates(self):
        urls = [f"https://example.com/{i}" for i in range(5)]
        assert collector_request(source_url=urls).source_url == urls
        with pytest.raises(ValidationError):
            collector_request(source_url=urls + ["https://example.com/5"])
        with pytest.raises(ValidationError):
            collector_request(source_url=["https://example.com/a", "https://example.com/a"])

    # ── the upload request ──
    @pytest.mark.parametrize(
        "name,mime",
        [("a.pdf", "application/pdf"), ("a.png", "image/png"), ("a.JPG", "image/jpeg"), ("notes.txt", "text/plain"),
         ("a.py", ""), ("a.docx", "application/octet-stream"),      # extension alone is enough, as in the composer
         ("noext", "text/plain")],                                   # ... and so is the MIME type
    )
    def test_uploads_the_composers_accept(self, name, mime):
        CollectorUploadRequest(filename=name, file_type=mime, file_size_bytes=10)

    @pytest.mark.parametrize(
        "name,mime,size",
        [("a.gif", "image/gif", 10), ("a.webp", "image/webp", 10), ("a.exe", "application/x-msdownload", 10),
         ("a.zip", "application/zip", 10), ("a.pdf", "application/pdf", 20 * 1024 * 1024 + 1),
         ("a.pdf", "application/pdf", 0), ("../a.pdf", "application/pdf", 10), ("d\\a.pdf", "application/pdf", 10),
         ("", "application/pdf", 10)],
    )
    def test_uploads_the_composers_refuse(self, name, mime, size):
        with pytest.raises(ValidationError):
            CollectorUploadRequest(filename=name, file_type=mime, file_size_bytes=size)

    def test_upload_requests_take_no_extra_fields(self):
        with pytest.raises(ValidationError):
            CollectorUploadRequest(filename="a.pdf", file_type="application/pdf", file_size_bytes=10, thread_id="x")

    def test_datetime_is_required(self):
        with pytest.raises(ValidationError):
            collector_request(personalization={"user_location": LOC})

    @pytest.mark.parametrize("model", ["rix", "gemini", "pro", "fast", "gpt-4"])
    def test_teacher_models_only(self, model):
        with pytest.raises(ValidationError):
            collector_request(model=model)

    @pytest.mark.parametrize("model", COLLECTOR_MODELS)
    def test_teacher_models_pass(self, model):
        collector_request(model=model)

    @pytest.mark.parametrize("q", ["", "   \n"])
    def test_blank_query(self, q):
        with pytest.raises(ValidationError):
            collector_request(query=q)

    def test_blank_memory_means_no_memory(self):
        assert collector_request(memory="  \n ").memory is None


# ── 2. identical to what /chat builds ──────────────────────────────────────


class TestSameContextAsChat:
    """`build_turn_context` is the function /chat calls; feeding it the payload the
    frontend sends and the request the collector builds must give the same bytes."""

    @pytest.mark.parametrize("lang", ["zh-CN", "en", "zh-TW", "ja", "ko", None])
    @pytest.mark.parametrize("loc", [LOC, "Tokyo, Japan (GPS Precise Location)", None])
    def test_system_reminder_matches(self, lang, loc):
        personalization = {"user_local_datetime": DT}
        if lang:
            personalization["response_language"] = lang
        if loc:
            personalization["user_location"] = loc
        ours = to_query_request(collector_request(personalization=personalization, memory=None), turn=1)
        theirs = QueryRequest(**frontend_payload(turn=1, with_memory=False, lang=lang, loc=loc))
        assert build_turn_context(ours, None)[0] == build_turn_context(theirs, None)[0]

    def test_first_turn_with_memory_matches(self):
        ours = to_query_request(collector_request(), turn=1)
        theirs = QueryRequest(**frontend_payload(turn=1))
        assert build_turn_context(ours, MEMORY) == build_turn_context(theirs, MEMORY)
        assert build_turn_context(ours, MEMORY)[1].endswith(MEMORY)

    @pytest.mark.parametrize("turn", [3, 5, 7])
    def test_memory_only_on_the_first_turn(self, turn):
        ours = to_query_request(collector_request(memory=None), turn=turn)
        theirs = QueryRequest(**frontend_payload(turn=turn))
        assert not memory_injection_due(ours) and not memory_injection_due(theirs)
        assert build_turn_context(theirs, MEMORY)[1] == ""

    def test_no_memory_means_no_block(self):
        ours = to_query_request(collector_request(memory=None), turn=1)
        assert build_turn_context(ours, None)[1] == ""

    def test_default_language_is_the_server_default(self):
        # production omits the field for "auto"; both then get Personalization's default
        ours = to_query_request(collector_request(personalization={"user_local_datetime": DT}), turn=1)
        assert "Response Language: Follow User's Query Language\n" in build_turn_context(ours, None)[0]

    def test_request_fields_match_the_frontend_payload(self):
        ours = to_query_request(collector_request(), turn=3)
        theirs = QueryRequest(**frontend_payload(turn=3))
        for field in ("query", "thread_id", "model", "turn"):
            assert getattr(ours, field) == getattr(theirs, field), field
        assert ours.personalization == theirs.personalization
        # nothing the collector doesn't model leaks in
        for field in ("skill", "source_url", "attached_file_ids", "follow_up_content"):
            assert getattr(ours, field) is None, field

    def test_files_and_urls_match_the_frontend_payload(self):
        # chat-view sends `attached_file_ids: [{id: name}]` and `source_url: [url]`, unmodified
        ours = to_query_request(
            collector_request(attached_file_ids=[{FID: "a.pdf"}], source_url=["https://example.com/"]), turn=1
        )
        theirs = QueryRequest(**{**frontend_payload(turn=1), "attached_file_ids": [{FID: "a.pdf"}], "source_url": ["https://example.com/"]})
        assert ours.attached_file_ids == theirs.attached_file_ids
        assert ours.source_url == theirs.source_url

    @pytest.mark.parametrize("skill", ["deep-research", "trip-advisor", "guided-learning"])
    def test_skill_matches_the_frontend_payload(self, skill):
        # chat-view sets `payload.skill = activeSkill` — the picker's id, unmodified
        ours = to_query_request(collector_request(skill=skill), turn=1)
        theirs = QueryRequest(**{**frontend_payload(turn=1), "skill": skill})
        assert ours.skill == theirs.skill == skill


@pytest.fixture(scope="module")
def build_message_content():
    # Deliberately not a skip when this import fails: a parity suite that
    # quietly stops checking the layout is worse than one that is red.
    for key, value in _IMPORT_ENV.items():
        os.environ.setdefault(key, value)
    from core.stream import build_message_content

    return build_message_content


# ── 3. the full user message ───────────────────────────────────────────────


class TestUserMessageLayout:
    """Render the user message the agent receives and compare it to a literal."""

    def render(self, build_message_content, request: QueryRequest, memory: str | None) -> str:
        reminder, user_memory = build_turn_context(request, memory)
        content, files, sources = build_message_content(
            request.query, reminder, None, request.thread_id, user_memory=user_memory
        )
        assert files == {} and sources == []
        assert isinstance(content, str)  # no images -> a plain string, as in production
        return content

    def test_first_turn_layout(self, build_message_content):
        ours = self.render(
            build_message_content, to_query_request(collector_request(), turn=1), MEMORY
        )
        theirs = self.render(build_message_content, QueryRequest(**frontend_payload(turn=1)), MEMORY)
        assert ours == theirs
        assert ours == (
            "<user_memory>\n"
            "Long-term facts about this user. Not all of it is relevant to the current turn.\n"
            f"{MEMORY}\n"
            "</user_memory>\n\n"
            "<system_reminder>\n"
            "You are Omni. If the user asks who you are, say you are Omni.\n"
            "Response Language: zh-CN\n"
            f"User Location: {LOC}\n"
            f"User Local Date Time: {DT}\n"
            "</system_reminder>\n\n"
            "<user_query>\n"
            "今天适合跑步吗？\n"
            "</user_query>"
        )

    @pytest.mark.parametrize(
        "picked,on_disk", [("deep-research", "web-research"), ("trip-advisor", "trip-advisor"), ("guided-learning", "guided-learning")]
    )
    def test_requested_skill_block(self, build_message_content, picked, on_disk):
        # /chat resolves the picker's id with resolve_skill_name before building the
        # message; the collector does the same, so the block names the on-disk skill.
        from core.agent import resolve_skill_name

        request = to_query_request(collector_request(skill=picked, memory=None), turn=1)
        reminder, user_memory = build_turn_context(request, None)
        content, _, _ = build_message_content(
            request.query, reminder, None, request.thread_id,
            user_memory=user_memory, skill=resolve_skill_name(request.skill),
        )
        assert f"</system_reminder>\n\n<requested_skill>\n{on_disk}\n</requested_skill>\n\n<user_query>" in content

    def test_later_turn_has_no_memory_block(self, build_message_content):
        ours = self.render(
            build_message_content, to_query_request(collector_request(memory=None), turn=3), None
        )
        theirs = self.render(build_message_content, QueryRequest(**frontend_payload(turn=3)), MEMORY)
        assert ours == theirs
        assert "<user_memory>" not in ours
