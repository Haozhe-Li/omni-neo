"""Thin wrapper around Fish Audio's streaming TTS WebSocket API.

Connection/auth setup still comes from the official `fish-audio-sdk`
(`fish_audio_sdk.AsyncWebSocketSession`), but `synthesize()` hand-rolls the
message loop instead of calling the SDK's own `.tts()` — verified against
the installed package (1.3.0) that its `tts()` only ever sends `StartEvent`,
then one `TextEvent` per chunk from our generator, then `CloseEvent`. It
never sends a `flush` event, and the SDK doesn't even define one
(`fish_audio_sdk.schemas` has no `FlushEvent` class), even though Fish's own
live-TTS docs (docs.fish.audio/api-reference/endpoint/websocket/tts-live)
describe `{"event": "flush"}` as exactly the mechanism for "force synthesis
of what's buffered right now, don't wait for more text." Without it, each
`TextEvent` just appends to the server's own buffer, which it only
auto-synthesizes once `chunk_length` (100-300 chars) is hit or the stream
closes — so a short lead-in sentence ("好的，我去帮你查一下天气") sent on
its own produces no audio at all until the *next* chunk arrives and pushes
the buffer over that threshold, or the turn ends. That's the whole point of
core/voice/prompt.py's "say something before calling a tool" rule getting
silently defeated: the lead-in text streams to the client and shows up in
the transcript, but no audio plays for it until the tool's result comes
back and gets appended too — i.e. TTS was never actually front-loading the
turn's latency, it just looked like it was because the text arrived early.

Sending a flush right after every chunk fixes this: each of
core/voice/session.py's already sentence/clause-sized chunks (see
`_drain_ready_chunks`) gets synthesized the moment it's ready, not batched
with whatever comes next.

Output is raw PCM16 at 24kHz mono — chosen over mp3/opus so the browser
client can schedule playback sample-accurately with plain Web Audio API
(`AudioBufferSourceNode`) instead of an MP3 decoder, which matters for
clean, immediate stop-on-barge-in.
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

import ormsgpack
from fish_audio_sdk import AsyncWebSocketSession, TTSRequest
from fish_audio_sdk.schemas import CloseEvent, StartEvent, TextEvent
from httpx_ws import aconnect_ws

SAMPLE_RATE = 24000

# The specific Fish Audio voice model to speak with, not whatever the backend
# would default to.
_REFERENCE_ID = "43a3b4034d564a11bdadee7c6f6c7039"

_FLUSH_EVENT = ormsgpack.packb({"event": "flush"})


class FishTTS:
    def __init__(self) -> None:
        api_key = os.environ.get("FISH_API_KEY")
        if not api_key:
            raise RuntimeError("FISH_API_KEY is not set")
        # Only used for its pre-authed httpx.AsyncClient (base_url + Bearer
        # header) and matching close() — see module docstring for why the
        # SDK's own .tts() isn't used for the actual message loop.
        self._session = AsyncWebSocketSession(api_key)

    async def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        """Feed sentence/clause chunks in as they're ready; PCM16 bytes come
        back progressively. One call = one Fish Audio WS connection, meant
        to span a whole agent turn (not re-opened per sentence)."""
        request = TTSRequest(
            text="",  # real text arrives via `text_stream`, per the SDK's own usage
            format="pcm",
            sample_rate=SAMPLE_RATE,
            # Fish's live-TTS docs mention a third "low" latency mode, but the
            # installed fish-audio-sdk's TTSRequest still only validates
            # Literal["normal", "balanced"] — confirmed live, "low" 400s the
            # request outright. Same lesson as before: check the installed
            # package, not just the docs. The flush-per-chunk fix above is
            # what actually matters for latency here; this field just picks
            # between its two real options.
            latency="balanced",
            reference_id=_REFERENCE_ID,
        )
        async with aconnect_ws(
            "/v1/tts/live", client=self._session._client, headers={"model": "speech-1.5"}
        ) as ws:

            async def sender() -> None:
                await ws.send_bytes(ormsgpack.packb(StartEvent(request=request).model_dump()))
                async for text in text_stream:
                    await ws.send_bytes(ormsgpack.packb(TextEvent(text=text).model_dump()))
                    await ws.send_bytes(_FLUSH_EVENT)
                await ws.send_bytes(ormsgpack.packb(CloseEvent().model_dump()))

            sender_task = asyncio.create_task(sender())
            try:
                while True:
                    message = await ws.receive_bytes()
                    data = ormsgpack.unpackb(message)
                    event = data["event"]
                    if event == "audio":
                        yield data["audio"]
                    elif event == "finish":
                        if data["reason"] == "error":
                            raise RuntimeError(f"Fish TTS error: {data}")
                        break
            finally:
                sender_task.cancel()
                try:
                    await sender_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001 — best-effort cleanup only
                    pass

    async def close(self) -> None:
        await self._session.close()
