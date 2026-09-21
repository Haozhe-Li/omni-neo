"""Per-connection orchestration: client audio -> Deepgram -> voice agent ->
Fish TTS -> client audio, with barge-in.

Turn lifecycle:
  1. Client streams continuous PCM16 audio (no client-side VAD for this MVP
     — see core/static/voice_test.html; Deepgram's own VAD does endpointing,
     one source of truth instead of two independently-tuned ones).
  2. Deepgram finalizes an utterance (`speech_final` or `UtteranceEnd`) ->
     `_finalize_user_turn` assigns a fresh `turn_id` and starts `_run_turn`.
  3. `_run_turn` streams the agent's reply, chunks it into sentences/clauses,
     feeds each chunk to a single Fish TTS connection spanning the whole
     turn, and forwards resulting PCM16 back to the client tagged with
     `turn_id` (8-byte header: turn_id, seq — both uint32 big-endian).
  4. If Deepgram's `SpeechStarted` fires while a turn is still in flight,
     that arms a barge-in — it does not fire one outright. Deepgram's own
     docs describe VAD as tonal-analysis speech-onset detection with no
     stated false-positive handling, and this app never stops streaming mic
     audio while the agent is talking (no client-side gating), so any echo
     of the agent's own TTS leaking back into the mic (routine without
     headphones) or ambient noise reliably fires SpeechStarted with no
     actual user speech behind it. Firing a cancel on that signal alone
     cuts the agent off mid-sentence, or — worse — cancels a turn within
     the first instant of `_finalize_user_turn`, which just looks like "I
     finished talking and nothing happened." So SpeechStarted only *arms*
     the barge-in; it's committed only once the next `Results` event
     carries actual recognized text, confirming a real utterance is
     underway. A genuine interruption still lands fast (Deepgram typically
     produces an interim hypothesis within ~100-300ms of true speech), a
     bare noise/echo blip that never resolves to words never cancels
     anything.
     Once committed, the turn task is cancelled and `active_turn_id` is
     bumped so any audio already in flight from the cancelled turn gets
     tagged stale and the client drops it on arrival — cancellation stops
     *new* chunks, the turn_id tag protects against ones already queued.
  5. `self.turn_task` finishing only means the server is done generating
     text and forwarding TTS audio — the client schedules playback ahead of
     time (see hooks/useVoiceSession.ts's nextPlayTimeRef), so several
     seconds of already-sent audio can still be audibly playing after
     turn_task completes. Gating "is the agent still speaking" on turn_task
     alone meant a real interruption landing in that tail window was
     silently ignored (SpeechStarted never armed, nothing ever cancelled
     the turn) — confirmed live, and worse for longer replies since the
     tail is proportionally bigger. `_turn_audible_until` tracks an exact
     estimate instead (summed straight from the PCM byte durations sent in
     tts_consumer, not a guess), and `_turn_is_audible` is what every
     barge-in check actually gates on now.
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import time

from fastapi import WebSocket

from core.database.db_user_usage import commit_charge_fast, evaluate_charge_fast
from core.utils.citations import all_citations, reset_citation_registry_async
from core.voice.agent import run_voice_turn
from core.voice.deepgram_stt import DeepgramSTT
from core.voice.fish_tts import SAMPLE_RATE, FishTTS
from core.voice.persist import persist_voice_turn

logger = logging.getLogger(__name__)

_TAIL_MARGIN_S = 0.3  # network/scheduling slack — see VoiceSession._turn_is_audible

_SENTENCE_END = set("。！？.!?\n")
_CLAUSE_END = set("，,、；;")
_SOFT_FLUSH_LEN = 14  # min length before a clause boundary alone triggers a flush
_HARD_FLUSH_LEN = 60  # flush regardless of punctuation past this length
_TRIVIAL_CHARS = _SENTENCE_END | _CLAUSE_END | set(" \t\r")


def _has_speakable_content(text: str) -> bool:
    """False for a chunk that's nothing but punctuation/whitespace.

    Splitting strictly on sentence/clause boundaries can hand back a chunk
    that's a single trailing character on its own — e.g. two sentence-enders
    back to back ("...。\n") cut the "。" into one chunk and leave a lone "\n"
    to start the next. core/voice/fish_tts.py now flushes every chunk
    immediately for latency (see its module docstring), and Fish's server
    hard-errors a flush with nothing to actually vocalize ("Inference
    backend returned empty audio"), which used to surface as the whole turn
    failing. Filtering here is what fixes that — the frontend transcript is
    unaffected either way, since agent_text is sent from the raw delta
    before it ever reaches this chunking.
    """
    return any(ch not in _TRIVIAL_CHARS for ch in text)


def _drain_ready_chunks(buffer: str, *, force: bool = False) -> tuple[list[str], str]:
    """Pull complete sentence/clause chunks off the front of `buffer`.

    Pure function so the chunking policy can be sanity-checked without a
    live LLM/TTS connection. With `force=True` (turn end, or a tool call
    about to fire), whatever's left becomes a final chunk too (or is simply
    dropped, if it has nothing speakable in it — force always clears
    `remaining` either way, since force means nothing carries over).
    """
    chunks: list[str] = []
    start = 0
    for i, ch in enumerate(buffer):
        length = i + 1 - start
        if ch in _SENTENCE_END or (ch in _CLAUSE_END and length >= _SOFT_FLUSH_LEN) or length >= _HARD_FLUSH_LEN:
            piece = buffer[start : i + 1]
            if _has_speakable_content(piece):
                chunks.append(piece)
            start = i + 1
    remaining = buffer[start:]
    if force:
        if remaining and _has_speakable_content(remaining):
            chunks.append(remaining)
        remaining = ""
    return chunks, remaining


class VoiceSession:
    def __init__(
        self,
        ws: WebSocket,
        thread_id: str,
        user_id: str,
        *,
        user_location: str | None = None,
        user_local_datetime: str | None = None,
    ) -> None:
        self.ws = ws
        self.thread_id = thread_id
        self.user_id = user_id
        self.user_location = user_location
        self.user_local_datetime = user_local_datetime
        self.stt = DeepgramSTT()  # raises if DEEPGRAM_API_KEY missing — fail fast, before accept-side cleanup gets messy
        self.active_turn_id = 0
        self.turn_task: asyncio.Task | None = None
        # monotonic deadline — see _turn_is_audible
        self._turn_audible_until = 0.0
        self._pending_transcript = ""
        # Set by a SpeechStarted event while a turn is in flight; only turns
        # into an actual barge-in once a subsequent Results event confirms
        # real recognized text (see module docstring point 4).
        self._barge_in_armed = False
        # agent_text/tool_call (text frames, from agent_loop) and TTS audio
        # (binary frames, from tts_consumer) are sent from two different
        # coroutines running concurrently under the same asyncio.gather —
        # harmless while audio only started flowing well after a turn's text
        # was mostly done, but the Fish TTS flush fix (core/voice/fish_tts.py)
        # made audio start almost immediately, so the two now genuinely
        # overlap in time. Two coroutines calling ws.send_*() at once on the
        # same Starlette WebSocket isn't safe — confirmed live, it started
        # throwing "unhandled errors in a TaskGroup" mid-turn as soon as text
        # and audio sends began overlapping for real. This lock serializes
        # every outbound frame (see _send_json and _send_audio below).
        self._ws_send_lock = asyncio.Lock()

    def _turn_is_audible(self) -> bool:
        """Whether the agent should still be treated as "speaking" for
        barge-in purposes — see module docstring point 5. True while the
        turn task is still generating/forwarding audio, OR while the
        client's already-sent audio should still be playing out per
        `_turn_audible_until`.
        """
        if self.turn_task is not None and not self.turn_task.done():
            return True
        return time.monotonic() < self._turn_audible_until + _TAIL_MARGIN_S

    async def run(self) -> None:
        await self.stt.connect()
        # Only now — Deepgram is actually connected and ready to receive —
        # tell the client it's safe to start streaming. Before this point
        # any audio the client already sent has been sitting unread in the
        # WebSocket's own receive buffer (accept() happens before this, so
        # the client's ws.onopen can fire early); the frontend buffers
        # locally until it sees this and flushes in order, so nothing said
        # in that gap gets lost, but the burst of already-arrived bytes was
        # still not going to Deepgram until now, which — if the frontend
        # had started streaming on its own — would have skewed Deepgram's
        # endpointing timing for the opening of the very first utterance.
        await self._send_json({"type": "ready"})
        deepgram_pump = asyncio.create_task(self._consume_deepgram_events())
        try:
            await self._pump_client_audio()
        finally:
            deepgram_pump.cancel()
            if self.turn_task is not None and not self.turn_task.done():
                self.turn_task.cancel()
            await self.stt.close()

    async def _pump_client_audio(self) -> None:
        while True:
            message = await self.ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("bytes")
            if data:
                self.stt.send(data)

    async def _consume_deepgram_events(self) -> None:
        async for event in self.stt.events():
            etype = event.get("type")
            if etype == "Results":
                alt = event["channel"]["alternatives"][0]
                text = alt.get("transcript", "")
                if not text:
                    continue
                if self._barge_in_armed:
                    self._barge_in_armed = False
                    # Re-check liveness rather than letting _barge_in's own
                    # _cancel_active_turn no-op silently: if the armed turn
                    # already finished (including its audible tail) in the
                    # meantime, this Results event belongs to an unrelated,
                    # perfectly normal later utterance — calling _barge_in()
                    # would still wipe _pending_transcript out from under it.
                    if self._turn_is_audible():
                        await self._barge_in()
                if event.get("is_final"):
                    self._pending_transcript = (self._pending_transcript + " " + text).strip()
                    await self._send_json({"type": "partial_transcript", "text": self._pending_transcript})
                    if event.get("speech_final"):
                        await self._finalize_user_turn()
                else:
                    live = (self._pending_transcript + " " + text).strip()
                    await self._send_json({"type": "partial_transcript", "text": live})
            elif etype == "UtteranceEnd":
                self._barge_in_armed = False
                if self._pending_transcript:
                    await self._finalize_user_turn()
            elif etype == "SpeechStarted":
                if self._turn_is_audible():
                    self._barge_in_armed = True

    async def _finalize_user_turn(self) -> None:
        text = self._pending_transcript.strip()
        self._pending_transcript = ""
        if not text:
            return
        # Gated before anything else — same evaluate-then-commit split /chat
        # uses (core/routers/chat.py): evaluate_charge_fast decides before a
        # turn starts, commit_charge_fast (fire-and-forget) only fires once
        # we're actually proceeding. Checked ahead of _cancel_active_turn so
        # a user who's out of credits doesn't lose whatever's still playing
        # from a turn they already paid for, just because they tried (and
        # failed) to start a new one. No turn_id on the error — same as a
        # missing-API-key rejection, this ends the whole call: every
        # following turn would fail the same gate.
        charge = await evaluate_charge_fast(self.user_id, "voice")
        if not charge["charged"]:
            await self._send_json({"type": "error", "detail": "Usage limit exceeded."})
            return
        commit_charge_fast(self.user_id, "voice")
        await self._cancel_active_turn()
        self.active_turn_id += 1
        turn_id = self.active_turn_id
        await self._send_json({"type": "turn_start", "turn_id": turn_id, "user_text": text})
        self.turn_task = asyncio.create_task(self._run_turn(turn_id, text))

    async def _cancel_active_turn(self) -> None:
        """Cancel whatever turn is in flight and tell the client, so stale
        audio/UI state never lingers into the next one.

        Shared by both ways an in-flight turn ends early: a confirmed
        barge-in, and a new utterance reaching speech_final/UtteranceEnd
        while the agent is still replying (_finalize_user_turn). Both mean
        "the user is talking now, whatever was playing is stale" — but only
        the barge-in path used to send `clear`. _finalize_user_turn's own
        cancellation was silent: the old turn's already-scheduled audio kept
        playing client-side with nothing telling it to stop, overlapping
        the new turn's speech instead of yielding to it.
        """
        if not self._turn_is_audible():
            return
        if self.turn_task is not None:
            self.turn_task.cancel()  # no-op if it already finished — the audible tail is what's left
        self._turn_audible_until = 0.0
        self.active_turn_id += 1
        await self._send_json({"type": "clear", "turn_id": self.active_turn_id})

    async def _barge_in(self) -> None:
        await self._cancel_active_turn()
        self._pending_transcript = ""

    async def _run_turn(self, turn_id: int, user_text: str) -> None:
        tts = FishTTS()
        text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._turn_audible_until = 0.0
        # Accumulated for persist_voice_turn once the turn completes — kept
        # separate from agent_loop's own `buffer`, which gets drained as TTS
        # chunks are cut from it and so never holds the full reply.
        full_text_parts: list[str] = []
        steps: list[dict] = []

        # Same registry every retrieval tool call feeds (core/utils/citations.py)
        # — reusing it here, the same way core/stream.py does for regular chat,
        # means web_search/fetch_url calls made from this turn are already
        # producing citation records; all_citations() below just reads them
        # back out once the turn is done. `turn_id` doubles as the "turn"
        # number — voice threads don't support rewind, so it only needs to be
        # monotonically increasing, not exactly aligned with anything else.
        await reset_citation_registry_async(self.thread_id, turn_id)

        async def text_stream():
            while True:
                chunk = await text_queue.get()
                if chunk is None:
                    return
                yield chunk

        async def emit_ready(chunks: list[str]) -> None:
            for chunk in chunks:
                await text_queue.put(chunk)

        async def agent_loop() -> None:
            buffer = ""
            async for event in run_voice_turn(
                self.thread_id,
                user_text,
                user_location=self.user_location,
                user_local_datetime=self.user_local_datetime,
            ):
                if turn_id != self.active_turn_id:
                    return  # superseded by a barge-in mid-generation
                etype = event["type"]
                if etype == "text":
                    buffer += event["delta"]
                    full_text_parts.append(event["delta"])
                    await self._send_json({"type": "agent_text", "turn_id": turn_id, "delta": event["delta"]})
                    ready, buffer = _drain_ready_chunks(buffer)
                    await emit_ready(ready)
                elif etype == "tool_start":
                    # Flush now: this is the model's spoken lead-in from the
                    # voice prompt ("嗯我去帮你查一查") and it needs to reach
                    # TTS before the tool's own latency, not after.
                    ready, buffer = _drain_ready_chunks(buffer, force=True)
                    await emit_ready(ready)
                    steps.append({
                        "tool": event["name"], "args": event.get("args", {}), "timestamp": int(time.time() * 1000),
                    })
                    await self._send_json(
                        {"type": "tool_call", "turn_id": turn_id, "tool": event["name"], "args": event["args"]}
                    )
                elif etype == "done":
                    ready, buffer = _drain_ready_chunks(buffer, force=True)
                    await emit_ready(ready)
            await text_queue.put(None)

        async def tts_consumer() -> None:
            seq = 0
            async for pcm_chunk in tts.synthesize(text_stream()):
                if turn_id != self.active_turn_id:
                    return
                seq += 1
                header = struct.pack(">II", turn_id, seq)
                await self._send_audio(header + pcm_chunk)
                # PCM16 mono — exact playback duration, not an estimate (see
                # _turn_is_audible).
                duration = len(pcm_chunk) / 2 / SAMPLE_RATE
                self._turn_audible_until = max(self._turn_audible_until, time.monotonic()) + duration

        try:
            await asyncio.gather(agent_loop(), tts_consumer())
            if turn_id == self.active_turn_id:
                await self._send_json({"type": "turn_end", "turn_id": turn_id})
                # Fire-and-forget: a Supabase round trip has no business
                # delaying turn_end (the client's cue to stop showing
                # "thinking"/"speaking"). A barge-in or a hangup that lands
                # before this finishes doesn't lose the write — it's not
                # tied to turn_task or the WS connection.
                asyncio.create_task(
                    persist_voice_turn(
                        self.thread_id, self.user_id, user_text, "".join(full_text_parts), steps, all_citations(),
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — surface pipeline failures to the client instead of a silent dropped task
            logger.exception("voice turn %s failed", turn_id)
            await self._send_json({"type": "error", "turn_id": turn_id, "detail": str(e)})
        finally:
            await tts.close()

    async def _send_json(self, payload: dict) -> None:
        try:
            async with self._ws_send_lock:
                await self.ws.send_text(json.dumps(payload, ensure_ascii=False))
        except RuntimeError:
            pass  # socket already closing

    async def _send_audio(self, data: bytes) -> None:
        try:
            async with self._ws_send_lock:
                await self.ws.send_bytes(data)
        except RuntimeError:
            pass  # socket already closing
