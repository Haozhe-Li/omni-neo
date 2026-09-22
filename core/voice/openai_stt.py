"""Thin wrapper around OpenAI's Realtime transcription WebSocket API.

Replaces core/voice/deepgram_stt.py (removed) — Deepgram's real-time config
has no streaming mode that reliably auto-picks between Chinese and English
(nova-3's codeswitch `language=multi` covers neither Chinese, and
`detect_language=true` isn't supported for streaming at all per Deepgram's own
docs), so it was pinned to `language=zh` and English speech came back as
noise. `gpt-live-transcribe`'s `languages` hint covers both in one stream —
verified live with a spoken English and a spoken Chinese sentence back to
back in the same session, both transcribed correctly.

Configured exactly as OpenAI's realtime-transcription guide recommends:
  - `gpt-live-transcribe`, which streams transcript deltas *while the user is
    still speaking* (measured: ~0.2-0.5s behind the audio). That is what
    live captions and barge-in both depend on. `gpt-transcribe` is the wrong
    model here even though it accepts server VAD: per the same guide it only
    starts transcribing after a turn is committed, and measured, every delta
    arrived in one burst ~0.7s after the speaker stopped — nothing at all
    while they talked.
  - `turn_detection: null`. The guide's recommended model returns its final
    transcript "when your application commits each audio turn", and the
    server rejects server_vad for it outright. Deciding when a turn ends —
    and sending `commit()` — is core/voice/session.py's job.
  - 24kHz PCM, as in every example in that guide. The frontend captures 16kHz
    (what Deepgram needed), so `_sender` upsamples 16k -> 24k with
    `audioop.ratecv`, state threaded across the whole call so chunk
    boundaries don't click. (`audioop` is gone in Python 3.13; the Docker
    image pins 3.12.)
  - Connected with `?intent=transcription`: the model page lists plain
    `v1/realtime` as unsupported for gpt-live-transcribe (`?model=...` there
    fails with invalid_model), and this is how a WebSocket lands on a
    transcription session instead.

Hand-rolled rather than an SDK, same reasoning as the file this replaces: a
JSON handshake (`session.update`), JSON-framed base64 audio
(`input_audio_buffer.append`) and commits in, JSON transcript events out.
"""
from __future__ import annotations

import asyncio
import audioop
import base64
import json
import os
from typing import AsyncIterator

import websockets

_OPENAI_REALTIME_WS_URL = "wss://api.openai.com/v1/realtime?intent=transcription"

_INPUT_RATE = 16000   # what the frontend captures and sends
_OPENAI_RATE = 24000

_SESSION_UPDATE = {
    "type": "session.update",
    "session": {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": _OPENAI_RATE},
                "transcription": {
                    "model": "gpt-live-transcribe",
                    "languages": ["en", "zh"],
                    "delay": "low",
                },
                "turn_detection": None,
            }
        },
    },
}

# Queued alongside audio so a commit lands after every chunk that arrived
# before it, never ahead of audio still waiting in the queue.
_COMMIT = object()


class OpenAISTT:
    """One connection for the lifetime of a voice session.

    `send()` and `commit()` are fire-and-forget — both only enqueue, and a
    background sender task does the resampling, base64/JSON framing and the
    actual network writes, in order. `events()` is the async-iterator side
    callers pump concurrently.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._api_key = api_key
        self._ws: websockets.ClientConnection | None = None
        self._send_queue: asyncio.Queue[object] = asyncio.Queue()
        self._sender_task: asyncio.Task | None = None
        # audioop.ratecv's continuation state, carried across every chunk of
        # the call (never reset per chunk) so the upsample stays seamless.
        self._resample_state = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            _OPENAI_REALTIME_WS_URL,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            max_size=None,
        )
        await self._ws.send(json.dumps(_SESSION_UPDATE))
        self._sender_task = asyncio.create_task(self._sender())

    async def _sender(self) -> None:
        assert self._ws is not None
        while True:
            item = await self._send_queue.get()
            if item is None:
                return
            if item is _COMMIT:
                payload = {"type": "input_audio_buffer.commit"}
            else:
                resampled, self._resample_state = audioop.ratecv(
                    item, 2, 1, _INPUT_RATE, _OPENAI_RATE, self._resample_state
                )
                payload = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(resampled).decode("ascii"),
                }
            try:
                await self._ws.send(json.dumps(payload))
            except websockets.exceptions.ConnectionClosed:
                return

    def send(self, pcm16_bytes: bytes) -> None:
        """Queue a raw 16kHz PCM16 mono frame. Never awaits or blocks."""
        self._send_queue.put_nowait(pcm16_bytes)

    def commit(self) -> None:
        """End the current audio turn: its final transcript arrives as a
        `conversation.item.input_audio_transcription.completed` event."""
        self._send_queue.put_nowait(_COMMIT)

    async def events(self) -> AsyncIterator[dict]:
        assert self._ws is not None
        async for raw in self._ws:
            if isinstance(raw, (bytes, bytearray)):
                continue  # this API doesn't send binary frames back, but be defensive
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                continue

    async def close(self) -> None:
        self._send_queue.put_nowait(None)
        if self._sender_task is not None:
            self._sender_task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except websockets.exceptions.ConnectionClosed:
                pass
