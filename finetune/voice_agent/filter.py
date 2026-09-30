"""Annotate collected traces with deterministic pass/fail flags and build the SFT file.

    python finetune/voice_agent/filter.py

Reads `dataset/traces.jsonl`. Writes, under the gitignored `dataset/`:

    annotated.jsonl   every record + `spoken`, `units`, `flags` (empty = clean)
    sft_train.jsonl   the clean records only, as {"messages", "tools"} rows

Nothing is dropped from `annotated.jsonl` — a failing record is *marked*, with
the reason, so a threshold can be revisited without re-collecting.

"Spoken text" is every assistant message's text in order (the lead-in before a
tool call plus the answer after it), because that is what the TTS reads out.
The checks, all deterministic:

  markup     a character the TTS reads badly (a line break is fine — TTS just
             pauses, and it is not flagged). Allowlist, not blocklist: letters,
             digits (any script), spaces, and a short set of plain punctuation
             (`. , ! ? ' " : ; - % $ °` and their Chinese counterparts). So
             brackets of every kind, markdown (`* # ` ~ _ | >`), slashes,
             `& + = @ ^`, emoji, and URLs all fail.
  too_long   more than 75 units, where a unit is one CJK character or one
             non-CJK word — "75 words" for English, "75 characters" for Chinese.
             (Was 50; the Chinese teacher answers ran 50-100 characters and 50
             cut 30 of 52 of them.)
  empty      nothing was said.

Three further flags are about the *record*, not the wording, and are kept
separate from the two requested rules so their counts can be read on their own:

  tool_error    a tool call failed, or the run errored — an answer built on a
                failed lookup is not one to imitate.
  identity      the reply says *it* was built by / is powered by / is another
                model or vendor (GPT, OpenAI, Gemma, ...) — a self-claim, not a
                mention (news about OpenAI is fine). The teacher is not told what powers Omni Voice, so on
                "what model are you" it can only guess, and imitating a guess
                teaches the student a false identity.
  no_leadin     a tool call with no spoken words before it in the same message
                (the prompt requires a lead-in; a silent call reads as a
                frozen connection, and it is what the student most needs to
                see done every time).
  no_answer     a non-hang-up trace that ends on a tool result / tool call.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "dataset"
# Flags that are annotated but do not keep a record out of sft_train.jsonl.
# markup: decided to leave to the downstream TTS cleaner (core/voice/tts_text.py
# already strips brackets, book-title marks and the like before speech), so the
# student may see them; they stay visible in annotated.jsonl and the report.
NON_BLOCKING = {"markup"}

MAX_UNITS = 75
END_CALL = "end_call"

_ALLOWED_PUNCT = set(".,!?'’\"“”‘:;-—–…%$°" + "，。！？、：；")
_CJK = re.compile(r"[⺀-鿿豈-﫿]")
_LATIN_WORD = re.compile(r"[^\W_]+(?:['’.\-,][^\W_]+)*")  # "don't", "3.5", "1,234", "well-known" = one word
_URL = re.compile(r"https?://|www\.", re.I)
_VENDOR = r"(?:gpt[-\w.]*|chatgpt|openai|luna|gemma|gemini|claude|anthropic|llama|qwen|mistral|deepseek|grok)"
# A claim about *itself*, not a mention: news that says "OpenAI launched ..."
# is the answer, "I was built by OpenAI" is the assistant misstating what it is.
_IDENTITY = re.compile(
    rf"\b(?:built|developed|trained|created|made|powered|driven|designed)\s+(?:by|on|with)\s+(?:the\s+)?{_VENDOR}\b"
    rf"|\bI(?:'m|’m|\s+am)\s+(?:a\s+|an\s+)?{_VENDOR}\b"
    rf"|(?:由|是|基于|用)\s*{_VENDOR}[^。，！？]{{0,8}}(?:开发|训练|打造|创建|制作|驱动|研发|模型)"
    rf"|我是\s*{_VENDOR}",
    re.I,
)
_LINE_BREAKERS = {" ": "\\u2028", " ": "\\u2029", "\u0085": "\\u0085"}


def read_jsonl(path: Path) -> list[dict]:
    # Split on "\n" only, never splitlines() — see finetune/pro_agent/build_dataset.py.
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def dump_row(row: dict) -> str:
    out = json.dumps(row, ensure_ascii=False)
    for ch, esc in _LINE_BREAKERS.items():
        out = out.replace(ch, esc)
    return out


def spoken_text(messages: list[dict]) -> tuple[str, list[str]]:
    """(the messages joined for measuring, the individual assistant texts)."""
    parts = [m["content"] for m in messages if m["role"] == "assistant" and m.get("content")]
    return "\n".join(parts), parts


def count_units(text: str) -> int:
    """CJK characters + non-CJK words. 50 is "50 words" in English, "50 characters" in Chinese."""
    return len(_CJK.findall(text)) + len(_LATIN_WORD.findall(_CJK.sub(" ", text)))


def bad_chars(text: str) -> list[str]:
    bad = {
        ch for ch in text
        if not (ch.isalnum() or ch == " " or ch in _ALLOWED_PUNCT or ch == "\n")
    }
    return sorted(bad)


def annotate(rec: dict) -> dict:
    flags: list[str] = []
    messages = rec["messages"]
    text, parts = spoken_text(messages)
    joined = " ".join(parts)

    if not joined.strip():
        flags.append("empty")
    bad = bad_chars(text)
    if bad or _URL.search(text):
        flags.append("markup:" + "".join(bad or ["url"]))
    units = count_units(joined)
    if units > MAX_UNITS:
        flags.append(f"too_long:{units}")

    if rec.get("error") or rec.get("tool_errors"):
        flags.append("tool_error")
    if _IDENTITY.search(joined):
        flags.append("identity")
    if any(m["role"] == "assistant" and m.get("tool_calls") and not m["content"].strip() for m in messages):
        flags.append("no_leadin")
    ends_call = any(c["name"] == END_CALL for c in rec.get("tool_calls", []))
    if messages and not ends_call and (messages[-1]["role"] != "assistant" or messages[-1].get("tool_calls")):
        flags.append("no_answer")

    return {**rec, "spoken": joined, "units": units, "flags": flags}


def split_lead_ins(messages: list[dict]) -> list[dict]:
    """[assistant(content + tool_calls)] -> [assistant(content), assistant(tool_calls)].

    Gemma 4's chat template does not render text that shares a message with a
    tool call where it was written: it moves it *after* the tool response
    (`<|tool_call>...<tool_response|>Let me check.It's sunny.`). Trained that
    way (omni-voice-v3), the model learned to say its "let me check" after the
    lookup, and to say nothing before it — a silent gap on a live call. Two
    consecutive assistant messages render as text-then-call
    (`Let me check.<|tool_call>...<tool_response|>It's sunny.`), which is what
    the served model has to produce, so that is the shape it is trained on.
    """
    out: list[dict] = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls") and m["content"].strip():
            out.append({"role": "assistant", "content": m["content"]})
            out.append({**m, "content": ""})
        else:
            out.append(m)
    return out


def sft_row(rec: dict) -> dict:
    messages = list(rec["messages"])
    # end_call is return_direct: the trace ends on the tool result, with no
    # assistant turn after it. What is being taught is the goodbye + the call,
    # already the last assistant message, so the trailing tool message goes.
    if messages[-1]["role"] == "tool":
        messages.pop()
    return {"messages": split_lead_ins(messages), "tools": rec["tools"]}


def main() -> int:
    records = read_jsonl(DATA / "traces.jsonl")
    annotated = [annotate(r) for r in records]

    with (DATA / "annotated.jsonl").open("w", encoding="utf-8") as f:
        for r in annotated:
            f.write(dump_row(r) + "\n")
    def blocking(r: dict) -> list[str]:
        return [fl for fl in r["flags"] if fl.split(":")[0] not in NON_BLOCKING]

    clean = [r for r in annotated if not blocking(r)]
    with (DATA / "sft_train.jsonl").open("w", encoding="utf-8") as f:
        for r in clean:
            f.write(dump_row(sft_row(r)) + "\n")

    kinds = Counter(fl.split(":")[0] for r in annotated for fl in r["flags"])
    print(f"{len(annotated)} records: {len(clean)} clean -> sft_train.jsonl, {len(annotated) - len(clean)} flagged")
    print("flag counts (a record can carry several):", dict(kinds) or "none")
    print("\nby language / category (clean / total):")
    for key in ("lang", "cat"):
        tot, ok = Counter(r[key] for r in annotated), Counter(r[key] for r in clean)
        print("  " + "  ".join(f"{k}={ok[k]}/{tot[k]}" for k in sorted(tot)))
    print("\nflagged (kept = annotated but still in sft_train.jsonl):")
    for r in annotated:
        if r["flags"]:
            tag = "kept" if not blocking(r) else "    "
            print(f"  {tag} {r['id']:<16} {' '.join(r['flags']):<26} {r['spoken'][:80]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
