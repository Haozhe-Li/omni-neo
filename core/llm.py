import os

from langchain_cerebras import ChatCerebras as _BaseChatCerebras
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
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
qwen_3_6_27b = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.2, max_completion_tokens=16384)
gemini_flash_lite_latest = init_chat_model("google_genai:gemini-flash-lite-latest")
gemini_flash = init_chat_model("google_genai:gemini-3-flash-preview", include_thoughts=True)
llama3_1_8b = ChatGroq(model="llama-3.1-8b-instant")
glm_4_7 = ChatCerebras(model="zai-glm-4.7", temperature=0.2)
gemma_4_31b = ChatCerebras(model="gemma-4-31b", temperature=0.2, reasoning_effort="low")
gemma_4_31b_high = ChatCerebras(model="gemma-4-31b", temperature=0.2, reasoning_effort="high")
prompt_guard_2_86m = ChatGroq(model="meta-llama/llama-prompt-guard-2-86m")
gemini_3_6_flash = init_chat_model("google_genai:gemini-3.6-flash")
gpt_5_6_luna = init_chat_model("openai:gpt-5.6-luna", use_responses_api=True)
gpt_5_6_terra = init_chat_model("openai:gpt-5.6-terra", use_responses_api=True)
gpt_5_4_mini = init_chat_model("openai:gpt-5.4-mini", use_responses_api=True)
gpt_5_4_nano = init_chat_model("openai:gpt-5.4-nano", use_responses_api=True)
qwen3_30b_a3b = ChatWandb(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507", temperature=0.2, max_tokens=8192
)
rix_30b_a3b_v1 = ChatWandb(
    model=(
        "wandb-artifact:///welogmediaofficial-university-of-illinois-urbana-champaign"
        "/omni-pro-agent/omni-pro-104-0812-0016:v1"
    ),
    temperature=0.7,
    max_tokens=8192,
)

# v3: 122 rows, 6 epochs, 732 steps, final loss 0.413. Superseded by v4 below;
# kept so the benchmark can still measure it.
#
# What changed from v1 (104 rows, 1 epoch, loss ~0.9): all 15 deep-research
# trajectories are back, and the run actually completed. Two earlier attempts at
# the full set hung forever because W&B Serverless Training has a hard
# **32,768-token** sequence limit that it does not error on — it just stops
# producing gradient steps while still reporting RUNNING. v2 died at step 97 of
# 516 on the first row above it. `filter_context.py` drops anything over 32,000
# measured through the real Qwen chat template; the longest row here is 31,840.
#
# temperature 0.2, not the 0.7 above: v1's published 0.866 was measured at 0.2,
# so this keeps the v1-vs-v3 comparison clean. Note that means re-running v1
# today would NOT reproduce 0.866 — its temperature was raised afterwards.
rix_30b_a3b_v3 = ChatWandb(
    model=(
        "wandb-artifact:///welogmediaofficial-university-of-illinois-urbana-champaign"
        "/omni-pro-agent/omni-pro-v3-0812-1157:v1"
    ),
    temperature=0.2,
    max_tokens=8192,
)

# v4: 157 rows, 6 epochs, 942 steps, final loss 0.422. Serves `rix`.
#
# The higher loss than v3's 0.413 is not a regression — the set grew 29% and
# every added row is a distribution v3 never saw. What v4 adds, all of it
# absent from the previous 147 rows:
#
#   - translation with a `<textblock>` deliverable (0 rows before, and 0 of the
#     28 benchmark cases, which is why v3 answers translations in plain prose)
#   - Chinese questions carrying English noun phrases. v3 answers those
#     entirely in English, 0 of 8 measured; one or two English words is fine at
#     8 of 8, so it is a dose response, not noise
#   - `Follow User's Query Language`, the `<personalization>` value production
#     sends when the user has set no preference. It appeared in 0 of 147 rows
#     while being what most production turns actually carry
#
# The 32,768-token trap that killed v2 is unchanged and still the thing to
# respect. v4's longest row is 29,128 measured through the real Qwen chat
# template with tool schemas included; nothing was dropped by the filter, and
# all 15 deep-research trajectories survive. Note that `build_dataset.py` had
# silently stopped water-filling tool results — without that fix the filter
# removes every deep-research row instead.
#
# temperature 0.2 to match v3, so the v3-vs-v4 benchmark comparison stays
# clean. Worth revisiting on its own: low temperature is what degenerate tool
# loops like best, and repeated identical calls are a live production
# complaint.
rix_30b_a3b_v4 = ChatWandb(
    model=(
        "wandb-artifact:///welogmediaofficial-university-of-illinois-urbana-champaign"
        "/omni-pro-agent/omni-pro-v4-0813-0317:v1"
    ),
    temperature=0.2,
    max_tokens=8192,
)

omni_widget_predictor_14b = ChatOpenAI(
    model=(
        os.environ["WIDGET_PREDICTOR_14B_MODEL"]
    ),
    base_url="https://api.inference.wandb.ai/v1",
    api_key=os.environ["WANDB_API_KEY"],
    temperature=0,
    max_tokens=128,
)

# For chat
chat_llm = gpt_oss_120b_low
vision_llm = gemma_4_31b

get_title_llm = gpt_oss_20b
prompt_guard_llm = prompt_guard_2_86m
update_memories_llm = gpt_oss_20b
widget_predictor_llm = omni_widget_predictor_14b
credibility_llm = gpt_oss_20b
generate_cover_llm = gpt_oss_20b
research_schedule_llm = gpt_oss_120b_low

CHAT_LLM_FALLBACKS = [gemma_4_31b, gemini_flash_lite_latest]
