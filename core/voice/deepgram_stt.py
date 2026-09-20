"""Thin wrapper around Deepgram's real-time streaming STT WebSocket API.

Hand-rolled rather than pulling in `deepgram-sdk`: the wire protocol (a
query-string config, raw PCM16 frames in, JSON transcript/VAD events out) is
small and stable, and a bare `websockets` connection is simpler to reason
about from `VoiceSession`'s barge-in path than an SDK-managed one.

Frontend must send 16-bit PCM, mono, 16kHz — that's baked into the query
string below (`encoding=linear16&sample_rate=16000&channels=1`); if the
client-side capture ever resamples differently, this URL has to change too.

Turn detection lives in Deepgram, not in a second, independently-tuned
front-end VAD: `endpointing=300` finalizes a transcript after 300ms of
silence, `utterance_end_ms=1000` + `vad_events=true` additionally emits
`UtteranceEnd` and `SpeechStarted` events off Deepgram's own VAD. Both are
consumed in `core/voice/session.py`.

`language=zh` rather than `multi` or `detect_language=true`: verified against
Deepgram's live docs (developers.deepgram.com/docs/models-languages-overview
and .../language-detection) that neither actually gives us "auto-detect
Chinese or English" in streaming. `language=multi` is nova-3's real-time
codeswitching mode, but its supported set is English/Spanish/French/German/
Hindi/Russian/Portuguese/Japanese/Italian/Dutch — Chinese is not in it, even
though nova-3 supports Chinese as a normal (non-codeswitch) language; that
mismatch is exactly why Chinese speech was coming back as English-ish noise.
`detect_language=true` does cover Chinese, but Deepgram's own docs say it
"is not currently supported for streaming" at all. So there is currently no
Deepgram streaming config that auto-picks between Chinese and English — this
pins to Chinese (nova-3 still tolerates short embedded English words fine in
practice), trading away reliable full-English utterances until/unless this
gets a proper per-session language-detection pass in front of the stream.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator

import websockets

_DEEPGRAM_WS_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model=nova-3&language=zh"
    "&encoding=linear16&sample_rate=16000&channels=1"
    "&interim_results=true&smart_format=true"
    "&endpointing=300&utterance_end_ms=1000&vad_events=true"
)


class DeepgramSTT:
    """One connection for the lifetime of a voice session.

    `send()` is fire-and-forget (queues audio for a background sender task);
    `events()` is the async-iterator side callers pump concurrently.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY is not set")
        self._api_key = api_key
        self._ws: websockets.ClientConnection | None = None
        self._send_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._sender_task: asyncio.Task | None = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(
            _DEEPGRAM_WS_URL,
            additional_headers={"Authorization": f"Token {self._api_key}"},
            max_size=None,
        )
        self._sender_task = asyncio.create_task(self._sender())

    async def _sender(self) -> None:
        assert self._ws is not None
        while True:
            chunk = await self._send_queue.get()
            if chunk is None:
                return
            try:
                await self._ws.send(chunk)
            except websockets.exceptions.ConnectionClosed:
                return

    def send(self, pcm16_bytes: bytes) -> None:
        """Queue a raw PCM16 audio frame. Safe to call from the hot path —
        never awaits, never blocks on the network."""
        self._send_queue.put_nowait(pcm16_bytes)

    async def events(self) -> AsyncIterator[dict]:
        assert self._ws is not None
        async for raw in self._ws:
            if isinstance(raw, (bytes, bytearray)):
                continue  # Deepgram doesn't send binary frames back, but be defensive
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
                await self._ws.send(json.dumps({"type": "CloseStream"}))
            except websockets.exceptions.ConnectionClosed:
                pass
            await self._ws.close()
