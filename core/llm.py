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

gpt_oss_120b_low_groq = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="low")
gpt_oss_120b_high_groq = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="high")
# reasoning_format="parsed": keeps reasoning tokens out of `content` (Groq's
# default "raw" inlines them as <think>...</think>) and puts them in
# additional_kwargs.reasoning_content instead, so core/stream.py can stream
# them as their own `reasoning` SSE event instead of leaking into the answer.
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
llama_guard_2 = ChatGroq(model="meta-llama/llama-prompt-guard-2-86m")

fast_llm = gpt_oss_120b_low
pro_llm = gemma_4_31b_high
get_title_llm = gpt_oss_20b
prompt_guard_llm = llama_guard_2
update_memories_llm = gpt_oss_20b
widget_predictor_llm = gpt_oss_20b
credibility_llm = gpt_oss_20b
generate_cover_llm = gpt_oss_20b
# Structured extraction only (title/instruction/schedule) — low reasoning
# effort is enough and keeps the interactive create-flow snappy.
research_schedule_llm = gpt_oss_120b_low
