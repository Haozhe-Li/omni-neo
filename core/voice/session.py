"""Per-connection orchestration: client audio -> OpenAI realtime STT -> voice
agent -> Fish TTS -> client audio, with barge-in.

Turn lifecycle:
  1. Client streams continuous PCM16 audio (no client-side VAD). The STT
     model (core/voice/openai_stt.py) has no turn detection of its own — it
     just streams transcript deltas while the user speaks and stops once
     they stop — so the end of a turn is read off that same delta stream:
     `_endpoint_watchdog` commits the audio turn once no new delta has
     arrived for `_ENDPOINT_SILENCE_S`. One signal decides both "what was
     said" and "they're done", instead of a second, separately-tuned audio
     VAD that could disagree with the transcript.
  2. The commit's `...transcription.completed` event carries the final text
     -> `_finalize_user_turn` assigns a fresh `turn_id` and starts
     `_run_turn`. If the user resumed speaking before that event landed,
     finalization waits for the new utterance too and both go out as one
     turn, rather than the agent answering half a thought.
  3. `_run_turn` streams the agent's reply, chunks it into sentences/clauses,
     feeds each chunk to a single Fish TTS connection spanning the whole
     turn, and forwards resulting PCM16 back to the client tagged with
     `turn_id` (8-byte header: turn_id, seq — both uint32 big-endian).
  4. The first transcript delta of a new utterance arriving while the agent
     is still audible is a barge-in. This app never stops streaming mic
     audio while the agent talks, so bare noise or an echo blip mustn't cut
     the agent off — but a delta is already recognized *text*, not a raw
     speech-onset guess, so it is itself the confirmation (there's no
     separate "armed" state to wait on). A genuine interruption still lands
     fast, since deltas trail live speech by only a few hundred ms.
     On barge-in, the turn task is cancelled and `active_turn_id` is
     bumped so any audio already in flight from the cancelled turn gets
     tagged stale and the client drops it on arrival — cancellation stops
     *new* chunks, the turn_id tag protects against ones already queued.
  5. `self.turn_task` finishing only means the server is done generating
     text and forwarding TTS audio — the client schedules playback ahead of
     time (see hooks/useVoiceSession.ts's nextPlayTimeRef), so several
     seconds of already-sent audio can still be audibly playing after
     turn_task completes. Gating "is the agent still speaking" on turn_task
     alone meant a real interruption landing in that tail window was
     silently ignored (nothing ever cancelled the turn) — confirmed live,
     and worse for longer replies since the
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
from core.voice.agent import END_CALL_TOOL_NAME, run_voice_turn
from core.voice.openai_stt import OpenAISTT
from core.voice.fish_tts import SAMPLE_RATE, FishTTS
from core.voice.persist import mark_voice_call_end, persist_voice_turn

logger = logging.getLogger(__name__)

_TAIL_MARGIN_S = 0.3  # network/scheduling slack — see VoiceSession._turn_is_audible

# No new transcript delta for this long ends the utterance. Measured against
# real speech: deltas trail the audio by ~0.2-0.5s, and the longest gap inside
# one sentence was 0.83s (a comma pause) — anything under ~1s splits
# sentences in half.
_ENDPOINT_SILENCE_S = 1.0
# A committed turn's completed event normally lands ~0.6s after the commit.
# Past this, finalize with the delta text already in hand rather than leave
# the user talking to a call that never answers.
_COMPLETION_TIMEOUT_S = 3.0
_WATCHDOG_TICK_S = 0.1

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
        self.stt = OpenAISTT()  # raises if OPENAI_API_KEY missing — fail fast, before accept-side cleanup gets messy
        self.active_turn_id = 0
        self.turn_task: asyncio.Task | None = None
        # monotonic deadline — see _turn_is_audible
        self._turn_audible_until = 0.0
        # Final text of committed utterances not yet sent to the agent.
        self._pending_transcript = ""
        # The uncommitted utterance currently receiving deltas. Deltas are
        # incremental fragments, so they're concatenated, keyed by the
        # item_id the STT server assigns to the open audio buffer.
        self._live_item_id: str | None = None
        self._live_text = ""
        self._last_delta_at = 0.0
        # Committed item_id -> (its delta text, commit time), until its
        # completed event arrives — see _endpoint_watchdog.
        self._awaiting: dict[str, tuple[str, float]] = {}
        # Every write this call makes to the thread's saved transcript, as one
        # chain (see _queue_persist): each read-modify-writes the same
        # ui_messages row, so two racing would drop one, and the call-ended
        # marker must land after the call's last turn, not beside it.
        self._persist_tail: asyncio.Task | None = None
        # ui_messages index of this call's latest saved reply — the call-start
        # marker goes on the first turn saved, the call-end one here.
        self._last_persisted_index: int | None = None
        self._call_end_written = False
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
        # Only now — the STT connection is actually up and ready to receive
        # — tell the client it's safe to start streaming. Before this point
        # any audio the client already sent has been sitting unread in the
        # WebSocket's own receive buffer (accept() happens before this, so
        # the client's ws.onopen can fire early); the frontend buffers
        # locally until it sees this and flushes in order, so nothing said
        # in that gap gets lost, but the burst of already-arrived bytes was
        # still not going anywhere until now, which — if the frontend had
        # started streaming on its own — would have arrived as one burst
        # rather than at speaking pace.
        await self._send_json({"type": "ready"})
        stt_pump = asyncio.create_task(self._consume_stt_events())
        watchdog = asyncio.create_task(self._endpoint_watchdog())
        try:
            await self._pump_client_audio()
        finally:
            stt_pump.cancel()
            watchdog.cancel()
            await self._cancel_turn_task()
            # A call that just dropped (no hangup message) still gets its
            # end marker. No-op after a clean hangup, which already wrote it.
            await self._write_call_end()
            await self.stt.close()

    async def _pump_client_audio(self) -> None:
        while True:
            message = await self.ws.receive()
            if message["type"] == "websocket.disconnect":
                return
            data = message.get("bytes")
            if data:
                self.stt.send(data)
                continue
            text = message.get("text")
            if text:
                try:
                    control = json.loads(text)
                except ValueError:
                    continue
                if control.get("type") == "hangup":
                    await self._hang_up()
                    return

    async def _cancel_turn_task(self) -> None:
        # Waited on, not just cancelled: until it has actually stopped, a turn
        # that already sent turn_end could still be about to queue its save.
        if self.turn_task is not None and not self.turn_task.done():
            self.turn_task.cancel()
            await asyncio.wait([self.turn_task])

    async def _hang_up(self) -> None:
        """The client's hang-up: finish writing the transcript, then ack.

        The client leaves for the thread's text view as soon as it hangs up,
        and would otherwise load it before the call-ended marker exists — so
        it waits for this ack (see useVoiceSession's hangUp) first.
        """
        await self._cancel_turn_task()
        await self._write_call_end()
        await self._send_json({"type": "hangup_ack"})

    def _queue_persist(self, user_text: str, agent_text: str, steps: list[dict], sources: list[dict]) -> None:
        """Save a finished turn after every save queued before it.

        Chained on the previous save's *task* rather than a lock: a save that
        has been created but hasn't started yet holds no lock, so a lock alone
        would let the call-end marker slip in ahead of it.
        """
        prev = self._persist_tail

        async def run() -> None:
            if prev is not None:
                await asyncio.wait([prev])
            index = await persist_voice_turn(
                self.thread_id, self.user_id, user_text, agent_text, steps, sources,
                call_start=self._last_persisted_index is None,
            )
            if index is not None:
                self._last_persisted_index = index

        self._persist_tail = asyncio.create_task(run())

    async def _write_call_end(self) -> None:
        if self._call_end_written:
            return
        self._call_end_written = True
        if self._persist_tail is not None:
            await asyncio.wait([self._persist_tail])
        # Nothing saved means no call worth marking in the transcript.
        if self._last_persisted_index is not None:
            await mark_voice_call_end(self.thread_id, self.user_id, self._last_persisted_index)

    async def _consume_stt_events(self) -> None:
        async for event in self.stt.events():
            etype = event.get("type")
            if etype == "conversation.item.input_audio_transcription.delta":
                await self._on_delta(event.get("item_id"), event.get("delta") or "")
            elif etype == "conversation.item.input_audio_transcription.completed":
                await self._on_completed(event.get("item_id"), event.get("transcript") or "")
            elif etype == "error":
                # A rejected session.update would otherwise leave the call
                # silently deaf — never transcribing, never erroring.
                logger.warning("voice STT error: %s", event.get("error"))

    async def _on_delta(self, item_id: str | None, delta: str) -> None:
        if not delta or item_id in self._awaiting:
            return  # a committed utterance's final text comes with its completed event
        if item_id != self._live_item_id:
            self._live_item_id = item_id
            self._live_text = ""
            # Module docstring point 4: the first words of a new utterance
            # while the agent is still audible are the barge-in.
            if self._turn_is_audible():
                await self._barge_in()
        self._live_text += delta
        self._last_delta_at = time.monotonic()
        await self._send_partial()

    async def _on_completed(self, item_id: str | None, transcript: str) -> None:
        entry = self._awaiting.pop(item_id, None)
        if entry is None:
            return  # already finalized from its delta text by the watchdog's timeout
        self._append_pending(transcript or entry[0])
        await self._send_partial()
        await self._maybe_finalize()

    async def _endpoint_watchdog(self) -> None:
        """Ends an utterance once its deltas go quiet (module docstring point
        1), and finalizes from delta text if a commit's completed event never
        shows up."""
        while True:
            await asyncio.sleep(_WATCHDOG_TICK_S)
            now = time.monotonic()
            if self._live_item_id is not None and now - self._last_delta_at >= _ENDPOINT_SILENCE_S:
                self._awaiting[self._live_item_id] = (self._live_text, now)
                self._live_item_id = None
                self._live_text = ""
                self.stt.commit()
            stale = [item_id for item_id, (_, at) in self._awaiting.items() if now - at >= _COMPLETION_TIMEOUT_S]
            for item_id in stale:
                text, _ = self._awaiting.pop(item_id)
                logger.warning("voice STT: no completed event for %s, finalizing from deltas", item_id)
                self._append_pending(text)
            if stale:
                await self._maybe_finalize()

    def _append_pending(self, text: str) -> None:
        text = text.strip()
        if text:
            self._pending_transcript = (self._pending_transcript + " " + text).strip()

    async def _maybe_finalize(self) -> None:
        # Hold off while the user is still mid-utterance or another commit is
        # in flight, so a resumed thought joins the same turn (point 2).
        if self._live_item_id is not None or self._awaiting:
            return
        await self._finalize_user_turn()

    async def _send_partial(self) -> None:
        text = (self._pending_transcript + " " + self._live_text).strip()
        if text:
            await self._send_json({"type": "partial_transcript", "text": text})

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

        Shared by both ways an in-flight turn ends early: a barge-in, and a
        new utterance being finalized while the agent is still replying
        (_finalize_user_turn). Both mean
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
                live_call=True,
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
                    # end_call is the client's cue to hang up (it still gets
                    # the tool_call below), not a step worth showing in the
                    # thread — the call-ended banner already says it.
                    if event["name"] != END_CALL_TOOL_NAME:
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
                # Not awaited: a Supabase round trip has no business delaying
                # turn_end (the client's cue to stop showing "thinking"/
                # "speaking"). A barge-in or a hangup that lands before this
                # finishes doesn't lose the write — it's not tied to
                # turn_task or the WS connection. Sources are read now, while
                # this turn's citation registry is still the current one.
                self._queue_persist(user_text, "".join(full_text_parts), steps, all_citations())
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
