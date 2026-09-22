"""Thin wrapper around OpenAI's Realtime transcription WebSocket API.

Replaces core/voice/deepgram_stt.py (removed) — Deepgram's real-time config
has no streaming mode that reliably auto-picks between Chinese and English
(nova-3's codeswitch `language=multi` covers neither Chinese, and
`detect_language=true` isn't supported for streaming at all per Deepgram's own
docs), so it was pinned to `language=zh` and English speech came back as
noise. `gpt-live-transcribe`'s `languages` hint (session config below) covers
both in the same stream, which is the entire reason for this swap.

Hand-rolled rather than an SDK, same reasoning as the file this replaces: the
wire protocol is a JSON handshake (`session.update`) followed by JSON-framed
audio (`input_audio_buffer.append`, base64 — unlike Deepgram, which took raw
binary frames directly) in, and JSON transcript/VAD events out.

Endpoint/header/event-name choices below are taken from OpenAI's realtime
transcription docs as of 2026-09; a couple of details (whether the connect
URL needs a `?model=` or `?intent=` query param the transcription-session
example didn't show, and the exact field name of anything not exercised by a
short utterance) could only be confirmed by verified docs excerpts, not a
live call — check the first real connection's `session.updated`/`error`
frames against this before trusting it in production.

Turn detection lives in OpenAI's own `server_vad`, not a second,
independently-tuned front-end VAD — same one-source-of-truth reasoning as the
file this replaces. `silence_duration_ms=500` is a starting point translated
from Deepgram's old `endpointing=300` + `utterance_end_ms=1000` pair (which
had no direct one-knob equivalent here); expect to retune against real calls.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import AsyncIterator

import websockets

_OPENAI_REALTIME_WS_URL = "wss://api.openai.com/v1/realtime"

_SESSION_UPDATE = {
    "type": "session.update",
    "session": {
        "type": "transcription",
        "audio": {
            "input": {
                # Matches the PCM16/mono/16kHz the frontend already captures
                # for Deepgram (core/voice/session.py's docstring) — no
                # frontend change needed, 16kHz is a supported rate here too.
                "format": {"type": "audio/pcm", "rate": 16000},
                "transcription": {
                    "model": "gpt-live-transcribe",
                    # The fix this whole module exists for: both languages in
                    # one stream, instead of Deepgram's pinned-to-one-language
                    # limitation. ISO 639-1 codes, per the docs.
                    "languages": ["en", "zh"],
                    # "low" = tuned for low-latency live captions, matching
                    # what this app already needs (live partial_transcript).
                    "delay": "low",
                },
                "turn_detection": {
                    "type": "server_vad",
                    "silence_duration_ms": 500,
                },
            }
        },
    },
}


class OpenAISTT:
    """One connection for the lifetime of a voice session.

    Same connect/send/events/close shape as the DeepgramSTT it replaces —
    core/voice/session.py only ever talked to that shape, not to
    Deepgram-specific details, so the event *payloads* `events()` yields are
    the only thing session.py's consumer needs to know changed.

    `send()` is fire-and-forget (queues audio for a background sender task
    that also does the base64 encoding this wire format requires); `events()`
    is the async-iterator side callers pump concurrently.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._api_key = api_key
        self._ws: websockets.ClientConnection | None = None
        self._send_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._sender_task: asyncio.Task | None = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            _OPENAI_REALTIME_WS_URL,
            additional_headers={"Authorization": f"Bearer {self._api_key}"},
            max_size=None,
        )
        # Must be the first thing sent — nothing is transcribed correctly
        # until the session knows type="transcription" plus the model/
        # languages/turn_detection above.
        await self._ws.send(json.dumps(_SESSION_UPDATE))
        self._sender_task = asyncio.create_task(self._sender())

    async def _sender(self) -> None:
        assert self._ws is not None
        while True:
            chunk = await self._send_queue.get()
            if chunk is None:
                return
            try:
                await self._ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }))
            except websockets.exceptions.ConnectionClosed:
                return

    def send(self, pcm16_bytes: bytes) -> None:
        """Queue a raw PCM16 audio frame. Safe to call from the hot path —
        never awaits, never blocks on the network. The base64/JSON framing
        this API requires happens in the sender task, not here."""
        self._send_queue.put_nowait(pcm16_bytes)

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
