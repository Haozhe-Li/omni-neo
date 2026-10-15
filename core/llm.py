import os

from langchain_cerebras import ChatCerebras as _BaseChatCerebras
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from core.think_strip import ThinkBlockStripper


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


class ChatWandb(ChatOpenAI):
    """W&B Inference — OpenAI-compatible wire format, but not OpenAI.

    Subclassed so the one place that cares can tell them apart:
    `evals/models.py` derives both the leaderboard label and the pricing key
    from the class, and a W&B-hosted Qwen filed under `openai` would be priced
    at GPT rates (30x off) and grouped into the wrong provider column.

    deepagents still resolves this to the `openai` harness profile —
    `ls_provider` is hardcoded in ChatOpenAI and inherited here, verified
    rather than assumed. That matters: an unregistered provider key would
    restore deepagents' `BASE_AGENT_PROMPT` and quietly change the system
    prompt for W&B models only, making their scores incomparable with
    everything else in the matrix.

    `os.environ[...]` deliberately, not `os.getenv`: ChatOpenAI silently falls
    back to OPENAI_API_KEY when handed None, which would send an OpenAI key to
    W&B and fail with a confusing 401 instead of naming the missing variable.
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(
            model=model,
            base_url="https://api.inference.wandb.ai/v1",
            api_key=os.environ["WANDB_API_KEY"],
            **kwargs,
        )

class ChatWandbGemma(ChatWandb):
    """A Gemma LoRA on W&B Inference, with its empty thinking block stripped.

    Gemma's generation prompt already ends with an empty thought block, which the
    training rendering never contains, so the served model emits a second
    `<|channel>thought\n<channel|>` at the start of every message it writes and
    W&B returns it as plain text (finetune/voice_agent/README.md, point 2). Left
    alone it shows in the answer and is stored in the thread, which the next turn
    then reads back.

    Filtered here, before the token callbacks fire, rather than in the stream
    layer: LangGraph streams from `on_llm_new_token`, so a filter downstream of
    `_astream` would run after the unfiltered text had already gone out. For that
    reason `run_manager` is withheld from the base class and called by hand.
    """

    @staticmethod
    def _clean_chunk(gen: ChatGenerationChunk, strip: ThinkBlockStripper) -> ChatGenerationChunk:
        msg = gen.message
        if isinstance(msg.content, str):
            msg.content = strip.feed(msg.content)
            gen.text = msg.content
        return gen

    @staticmethod
    def _tail(strip: ThinkBlockStripper) -> ChatGenerationChunk | None:
        rest = strip.flush()
        return ChatGenerationChunk(message=AIMessageChunk(content=rest)) if rest else None

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        strip = ThinkBlockStripper()
        async for gen in super()._astream(messages, stop, None, **kwargs):
            gen = self._clean_chunk(gen, strip)
            if run_manager:
                await run_manager.on_llm_new_token(gen.text, chunk=gen)
            yield gen
        if (tail := self._tail(strip)) is not None:
            if run_manager:
                await run_manager.on_llm_new_token(tail.text, chunk=tail)
            yield tail

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        strip = ThinkBlockStripper()
        for gen in super()._stream(messages, stop, None, **kwargs):
            gen = self._clean_chunk(gen, strip)
            if run_manager:
                run_manager.on_llm_new_token(gen.text, chunk=gen)
            yield gen
        if (tail := self._tail(strip)) is not None:
            if run_manager:
                run_manager.on_llm_new_token(tail.text, chunk=tail)
            yield tail

    @staticmethod
    def _clean_result(result):
        for g in result.generations:
            if isinstance(g.message.content, str):
                strip = ThinkBlockStripper()
                g.message.content = strip.feed(g.message.content) + strip.flush()
                g.text = g.message.content
        return result

    def _generate(self, *args, **kwargs):
        return self._clean_result(super()._generate(*args, **kwargs))

    async def _agenerate(self, *args, **kwargs):
        return self._clean_result(await super()._agenerate(*args, **kwargs))


    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        """Drop the per-chunk usage report W&B sends on every single chunk.

        W&B repeats a *cumulative* usage object on all of them (measured: 29 of
        29 chunks on a 27-token completion), and LangChain's chunk `+` operator
        sums `usage_metadata` as it assembles the message — so a streamed reply
        arrives claiming roughly `n_chunks x` its real token count. That is not
        a rounding error: an eval case with a 7.5k prompt reported 94,391 input
        tokens, which then flows straight into `eval_results.cost_usd`.

        Exactly one chunk carries empty `choices` — the terminal usage-only one
        — and its cumulative figure is the true total. Dropping usage from every
        chunk that still has `choices` leaves that one intact, so the assembled
        message ends up with the correct number instead of a sum of prefixes.

        Overriding this single hook covers sync and async both: `_stream` and
        `_astream` funnel every chunk through it (the same reason the
        ChatCerebras override above sits here rather than in `_astream`).
        """
        if chunk.get("choices") and chunk.get("usage") is not None:
            chunk = {**chunk, "usage": None}
        return super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )


gpt_oss_120b_low = ChatCerebras(model="gpt-oss-120b", temperature=0.2, reasoning_effort="low")
gpt_oss_120b_high = ChatCerebras(model="gpt-oss-120b", temperature=0.2, reasoning_effort="high")
gpt_oss_120b_medium = ChatCerebras(model="gpt-oss-120b", temperature=0.2, reasoning_effort="medium")
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
gpt_oss_20b_low = ChatGroq(
    model="openai/gpt-oss-20b", temperature=0.1, reasoning_effort="low"
)
qwen_3_6_27b = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.2, max_completion_tokens=16384)
gemini_flash_lite_latest = init_chat_model("google_genai:gemini-flash-lite-latest")
gemini_flash = init_chat_model("google_genai:gemini-3-flash-preview", include_thoughts=True)
llama3_1_8b = ChatGroq(model="llama-3.1-8b-instant")
glm_4_7 = ChatCerebras(model="zai-glm-4.7", temperature=0.2)
prompt_guard_2_86m = ChatGroq(model="meta-llama/llama-prompt-guard-2-86m")
gemini_3_6_flash = init_chat_model("google_genai:gemini-3.6-flash")
gpt_6_luna = init_chat_model("openai:gpt-6-luna", use_responses_api=True)
# The voice agent's model (core/voice/agent.py): reasoning fully off and terse
# output, on the Fast service tier — every token is spoken aloud to someone
# waiting on the line, so time-to-first-token is the whole game.
gpt_6_luna_voice = init_chat_model(
    "openai:gpt-6-luna",
    use_responses_api=True,
    verbosity="low",
    reasoning={"effort": "none", "mode": "standard", "summary": None},
    service_tier="fast",
)
gpt_5_6_terra = init_chat_model("openai:gpt-5.6-terra", use_responses_api=True)
gpt_5_4_mini = init_chat_model("openai:gpt-5.4-mini", use_responses_api=True)
gpt_5_4_nano = init_chat_model("openai:gpt-5.4-nano", use_responses_api=True)
qwen3_30b_a3b = ChatWandb(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507", temperature=0.2, max_tokens=8192
)
# The chat fine-tune: Gemma 4 26B-A4B distilled from gpt-6-luna (finetune/rix_gemma).
# v1: 124 rows, 3 epochs, 372 steps, per-epoch mean loss 1.91 -> 0.94 -> 0.78.
# Rows come from thumbs-ups and the collector page, and include a handful of
# image turns, so unlike the Qwen adapters it replaces this one is served with
# vision. Trained on the current harness (harness_hash 0417ee59ecf80b04): a
# change to the system prompt, a skill's `description:` or a tool docstring
# means re-collecting and re-training, not just editing the prompt.
#
# Known gap from the first check against the full agent: it loads a skill
# (read_file on /skills/...) far less often than luna does. The empty thought
# marker it emits on every message is stripped by `ChatWandbGemma`.
#
# temperature 0.2, as every earlier rix.
rix_gemma_v1 = ChatWandbGemma(
    model=(
        "wandb-artifact:///welogmediaofficial-university-of-illinois-urbana-champaign"
        "/omni-rix-gemma/rix-gemma-v1:v1"
    ),
    temperature=0.2,
    max_tokens=8192,
)

# The voice fine-tune: Gemma 4 26B-A4B distilled from gpt-6-luna on the live-call
# voice agent (finetune/voice_agent). v4: 143 rows, 3 epochs, 429 steps.
# Trained on the live-call system prompt and tool schemas
# (`VOICE_SYSTEM_PROMPT + VOICE_CALL_PROMPT_ADDENDUM`, with end_call), and on
# hand-written answers for identity ("who built you", "what model are you",
# "are you ChatGPT") — so a change to that prompt or those tools means
# re-collecting and re-training, not just editing core/voice/prompt.py.
#
# Known gap: English tool calls still arrive with no words before them, and the
# "let me check" comes after the result instead. Cause is not the data (v4
# splits the lead-in into its own assistant message so Gemma's chat template
# renders it before the call) — Gemma's generation prompt already ends in an
# empty thought block that the training rendering never contains. See
# finetune/voice_agent/README.md.
#
# max_tokens is small on purpose: every reply is spoken and the training set
# caps them at 75 words/characters.
rix_voice = ChatWandb(
    model=(
        "wandb-artifact:///welogmediaofficial-university-of-illinois-urbana-champaign"
        "/omni-voice-agent/omni-voice-v4:v1"
    ),
    temperature=0.2,
    max_tokens=1024,
)

# The model the pro agent is served on, as the eval matrix and
# finetune/pro_agent/fingerprint.py read it, and the fallback for an image turn
# on `best`. `best` itself serves luna on both paths for now
# (core/chat_models.py), so `chat_llm` names the fine-tune rather than following
# that switch: the fingerprint resolves its harness profile from the model's
# provider, and pointed at an OpenAI model it would capture a system prompt the
# adapter is never served.
chat_llm = rix_gemma_v1
vision_llm = gpt_6_luna

get_title_llm = gpt_oss_20b
prompt_guard_llm = prompt_guard_2_86m
update_memories_llm = gpt_oss_20b
# One structured-output call in front of every turn — see core/context_enrichment.py.
context_enrich_llm = gpt_oss_20b_low
credibility_llm = gpt_oss_20b
generate_cover_llm = gpt_oss_20b
research_schedule_llm = gpt_oss_120b_low

# Tried in order when the picked model errors. `gemma-4-31b` led this list until
# Cerebras archived it (404 `model_archived_error`), which left the first
# fallback dead on every failover.
CHAT_LLM_FALLBACKS = [gemini_flash_lite_latest]
