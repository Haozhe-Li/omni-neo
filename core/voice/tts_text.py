"""Clean a chunk of the agent's reply before it is spoken.

The voice prompt tells the model to write speech, not markdown, and
gpt-oss-120b follows that unevenly — bold, bullets, headings, links and the
occasional table row still come through, and Fish reads them out as literal
symbols ("asterisk asterisk"). This strips the writing, not the words.

Deliberately NOT a markdown parser: this runs on the sentence/clause chunks
core/voice/session.py cuts out of a live stream, so a `**span**` or a list
item routinely arrives split across two chunks and no parser would see a
balanced document. Dropping syntax *characters* needs no matching pair and so
survives any split; only the few constructs that must be read as a whole
(links, table rows) are patterns, and those simply fall back to dropping
their punctuation when a chunk cuts through the middle of one.

Brackets keep their contents — "16 °C (about 61 °F)" is spoken in full,
minus the parentheses, since the aside is usually part of the answer.
"""
from __future__ import annotations

import re

# [label](url) -> label, before brackets are stripped individually.
_MD_LINK = re.compile(r"\[([^\]\n]*)\]\(\s*<?[^)\n]*>?\s*\)")
_BARE_URL = re.compile(r"(?:https?://|www\.)\S+")
# Debris from a URL a chunk boundary cut through before this ever saw it —
# a domain, or a multi-segment path. Both are narrow on purpose: "m/s" has one
# slash and no domain, so it still gets spoken.
_URL_DEBRIS = re.compile(r"\S*\.(?:com|org|net|io|gov|edu|co|cn|ai|dev)\b\S*|\S*/\S+/\S*", re.I)
# A whole line that is just a table row or rule — there is nothing in
# "| --- | --- |" worth saying.
_TABLE_LINE = re.compile(r"^[ \t]*\|.*$|^[ \t]*[-=*_]{3,}[ \t]*$", re.M)
# Leading markers: #, >, bullets. Numbered items keep their number ("1." reads
# fine), so they are not listed here.
_LINE_MARKER = re.compile(r"^[ \t]*(?:#{1,6}|>+|[-*+•·])[ \t]+", re.M)
# Emoji and the symbol/arrow/dingbat blocks around them.
_EMOJI = re.compile(
    "[\U0001f000-\U0001faff\U0001f900-\U0001f9ff☀-➿←-⇿⬀-⯿️‍]"
)
# Everything left that is punctuation for the eye only. Brackets included —
# their contents are kept (see the module docstring).
# `$` is deliberately kept: "$12.99" is far more common in a spoken answer
# than LaTeX, and the backslashes of \( \) go with the rest of the escapes.
_SYNTAX_CHARS = str.maketrans("", "", "*`~#>|[]{}()（）【】《》〈〉<>\\")
_WHITESPACE = re.compile(r"\s+")


def normalize_for_tts(text: str) -> str:
    """Return `text` with the markup removed, ready to be spoken.

    An empty string means the chunk was nothing but markup — the caller
    should skip it rather than flush it, since Fish errors on a flush with
    nothing to vocalize (see core/voice/session.py's _has_speakable_content).
    """
    text = _MD_LINK.sub(r"\1", text)
    text = _BARE_URL.sub(" ", text)
    text = _URL_DEBRIS.sub(" ", text)
    text = _TABLE_LINE.sub(" ", text)
    text = _LINE_MARKER.sub("", text)
    text = _EMOJI.sub("", text)
    text = text.translate(_SYNTAX_CHARS)
    # Underscores only where they're emphasis (_word_, __word__), never inside
    # an identifier like file_name, which should still be read as one word.
    text = re.sub(r"(?<![\w])_{1,2}(?=\S)|(?<=\S)_{1,2}(?![\w])", "", text)
    return _WHITESPACE.sub(" ", text).strip()
