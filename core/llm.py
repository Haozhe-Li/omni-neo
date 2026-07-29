from langchain_cerebras import ChatCerebras as _BaseChatCerebras
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model


class ChatCerebras(_BaseChatCerebras):
    """langchain_cerebras (0.8.2) only extracts the delta's `reasoning` field
    into additional_kwargs inside its sync `_stream` override. Async calls
    (agent.astream, used for every request in this app) fall through to
    BaseChatOpenAI._astream, which has no Cerebras-specific handling, so
    reasoning silently vanished on every real request regardless of the
    `reasoning`/`reasoning_content` key lookup in core/stream.py. Both
    `_stream` and `_astream` funnel every chunk through
    `_convert_chunk_to_generation_chunk`, so overriding that single hook
    fixes both call paths instead of re-overriding `_astream` separately.
    """

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return generation_chunk
        choices = chunk.get("choices", [])
        if choices:
            reasoning = choices[0].get("delta", {}).get("reasoning")
            if reasoning:
                generation_chunk.message.additional_kwargs["reasoning"] = reasoning
        return generation_chunk


gpt_oss_120b_low = ChatCerebras(model="gpt-oss-120b", temperature=0.2, reasoning_effort="low")
gpt_oss_120b_high = ChatCerebras(model="gpt-oss-120b", temperature=0.2, reasoning_effort="high")
gpt_oss_120b_medium = ChatCerebras(model="gpt-oss-120b", temperature=0.2, reasoning_effort="medium")

# reasoning_format="parsed": keeps reasoning tokens out of `content` (Groq's
# default "raw" inlines them as <think>...</think>) and puts them in
# additional_kwargs.reasoning_content instead, so core/stream.py can stream
# them as their own `reasoning` SSE event instead of leaking into the answer.
# Set on every Groq gpt-oss instance, not just the one that happened to need it
# first: any of these can end up serving a chat turn via the fallback chain
# below, and a raw <think> block landing in the answer stream is a visible bug.
gpt_oss_120b_low_groq = ChatGroq(
    model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="low", reasoning_format="parsed"
)
gpt_oss_120b_high_groq = ChatGroq(
    model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="high", reasoning_format="parsed"
)
gpt_oss_120b_medium_groq = ChatGroq(
    model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="medium", reasoning_format="parsed"
)

gpt_oss_20b = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
qwen_3_6_27b = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.2, max_completion_tokens=16384)
gemini_flash_lite_latest = init_chat_model("google_genai:gemini-flash-lite-latest")
gemini_flash = init_chat_model("google_genai:gemini-3-flash-preview", include_thoughts=True)
llama3_1_8b = ChatGroq(model="llama-3.1-8b-instant")
# glm_4_7 = ChatCerebras(model="zai-glm-4.7", temperature=0.2)
gemma_4_31b = ChatCerebras(model="gemma-4-31b", temperature=0.2, reasoning_effort="low")
gemma_4_31b_high = ChatCerebras(model="gemma-4-31b", temperature=0.2, reasoning_effort="high")
# NOT the content-moderation "Llama Guard" family despite the model id's
# "guard" naming similarity — this is Prompt Guard 2 86M, a small classifier
# purpose-built to score a span of text for prompt-injection/jailbreak intent,
# with a real 512-token context window (core/prompt_guard.py's is_harmful
# truncates to that budget before calling it).
prompt_guard_2_86m = ChatGroq(model="meta-llama/llama-prompt-guard-2-86m")

fast_llm = gpt_oss_120b_low
pro_llm = gemma_4_31b_high
get_title_llm = gpt_oss_20b
prompt_guard_llm = prompt_guard_2_86m
update_memories_llm = gpt_oss_20b
widget_predictor_llm = gpt_oss_20b
credibility_llm = gpt_oss_20b
generate_cover_llm = gpt_oss_20b
# Structured extraction only (title/instruction/schedule) — low reasoning
# effort is enough and keeps the interactive create-flow snappy.
research_schedule_llm = gpt_oss_120b_low

# ── Chat fallbacks ──────────────────────────────────────────────────────────
# Cerebras hosts both interactive models and has been flaky, so each chat
# profile carries an ordered fallback chain, tried left to right by
# ModelFallbackMiddleware when the primary call raises (see core/agent.py).
# Only the two interactive profiles have one: everything else here is a
# short, non-interactive call where a failure degrades one feature rather than
# breaking the answer the user is waiting on.
#
# fast: the same gpt-oss-120b at the same reasoning effort, hosted by Groq —
# identical behaviour, different provider. Gemini Flash-Lite backs it up as a
# last resort AND as the multimodal escape hatch: the fast profile swaps to a
# vision model when an image is in the conversation, and if Cerebras is the
# thing that's down, neither Cerebras Gemma nor text-only gpt-oss can serve
# that turn. The doomed Groq attempt costs one failed request on the rare
# image-plus-outage overlap, which beats special-casing the chain per request.
#
# pro: Gemini Flash-Lite, the same model the scheduled agent already runs on.
FAST_LLM_FALLBACKS = [gpt_oss_120b_low_groq, gemini_flash_lite_latest]
PRO_LLM_FALLBACKS = [gemini_flash_lite_latest]
