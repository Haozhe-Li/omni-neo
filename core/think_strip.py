"""Strip Gemma's empty thinking block from model output.

Shared by the voice agent and the chat fine-tune (core/llm.py `ChatWandbGemma`):
both are Gemma LoRAs served by W&B, and both emit the marker the generation
prompt already ends with. See finetune/voice_agent/README.md, point 2.
"""
from __future__ import annotations


class ThinkBlockStripper:
    """Drop Gemma's empty thinking block from streamed text.

    The Gemma-based voice fine-tune (core/llm.py `rix_voice`, and `rix_gemma_v1` for chat) opens a
    thinking channel before its answer, and W&B serves it as plain text, with
    the special tokens surviving unpredictably. Three shapes were seen at the
    start of a model message: `<|channel>thought\n<channel|>`, a bare
    `thought\n<channel|>` when the opener is swallowed, and `:thought\n` with
    no closer at all. Unfiltered it would be spoken, and on a live call
    `_LEAK_START` would treat a `<` in it as a leak and cut the reply off.

    Only the explicit `<|channel>` form is treated as a block that runs to its
    closer. The other two strip just the `thought\n` label (and a closer, if one
    follows), so a missing closer can never swallow the answer.

    Markers arrive split across chunks, so a possible partial marker is held
    back. The label forms are only recognised at the *start* of a model message
    (`new_message()` marks each one), so an answer that merely contains the word
    "thought" is untouched. One instance per turn.
    """

    OPEN, CLOSE = "<|channel>", "<channel|>"
    LABELS = (":thought\n", "thought\n")

    def __init__(self) -> None:
        self._buf = ""
        # head: start of a model message | label: just after a `thought\n` label
        # | normal | inside: within an explicit <|channel> ... <channel|> block
        self._mode = "head"

    def new_message(self) -> str:
        """A new model message begins; returns any text still held from the last."""
        out = self.flush()
        self._mode = "head"
        return out

    def feed(self, delta: str) -> str:
        self._buf += delta
        out = ""
        while True:
            if self._mode == "head":
                if self._buf.startswith(self.OPEN):
                    self._buf, self._mode = self._buf[len(self.OPEN):], "inside"
                    continue
                label = next((l for l in self.LABELS if self._buf.startswith(l)), None)
                if label:
                    self._buf, self._mode = self._buf[len(label):], "label"
                    continue
                if any(c.startswith(self._buf) for c in (self.OPEN, *self.LABELS)):
                    return out  # could still become one of them — wait
                self._mode = "normal"
                continue
            if self._mode == "label":
                if self._buf.startswith(self.CLOSE):
                    self._buf = self._buf[len(self.CLOSE):]
                elif self.CLOSE.startswith(self._buf):
                    return out  # empty so far, or a partial closer — wait
                self._mode = "normal"
                continue
            if self._mode == "inside":
                i = self._buf.find(self.CLOSE)
                if i >= 0:
                    self._buf, self._mode = self._buf[i + len(self.CLOSE):], "normal"
                    continue
                self._buf = self._buf[len(self._buf) - self._hold(self.CLOSE):]
                return out
            # normal: cut out an opener anywhere; a stray closer is just dropped.
            hit = min((i for i in (self._buf.find(self.OPEN), self._buf.find(self.CLOSE)) if i >= 0), default=-1)
            if hit >= 0:
                out += self._buf[:hit]
                if self._buf.startswith(self.OPEN, hit):
                    self._buf, self._mode = self._buf[hit + len(self.OPEN):], "inside"
                else:
                    self._buf = self._buf[hit + len(self.CLOSE):]
                continue
            keep = max(self._hold(self.OPEN), self._hold(self.CLOSE))
            out += self._buf[: len(self._buf) - keep]
            self._buf = self._buf[len(self._buf) - keep:]
            return out

    def _hold(self, marker: str) -> int:
        """Length of the longest buffer tail that is a proper prefix of `marker`
        — what has to be kept back in case the next chunk completes it."""
        return next((k for k in range(min(len(marker) - 1, len(self._buf)), 0, -1)
                     if marker.startswith(self._buf[-k:])), 0)

    def flush(self) -> str:
        out = "" if self._mode == "inside" else self._buf
        self._buf = ""
        return out
