from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model

gpt_oss_120b_low = ChatCerebras(model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="low")
gpt_oss_120b_high = ChatCerebras(model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="high")
gpt_oss_120b_medium = ChatCerebras(model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="medium")
gpt_oss_20b = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
qwen_3_6_27b = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.2, max_completion_tokens=16384)
gemini_flash_lite_latest = init_chat_model("google_genai:gemini-flash-lite-latest")
gemini_flash = init_chat_model("google_genai:gemini-3-flash-preview")
llama3_1_8b = ChatGroq(model="llama-3.1-8b-instant")
glm_4_7 = ChatCerebras(model="zai-glm-4.7", temperature=0.2)
gemma_4_31b = ChatCerebras(model="gemma-4-31b", temperature=0.2)
llama_guard_2 = ChatGroq(model="meta-llama/llama-prompt-guard-2-86m")

fast_llm = gpt_oss_120b_high
pro_llm = gemini_flash
get_title_llm = gpt_oss_20b
prompt_guard_llm = llama_guard_2
update_memories_llm = gpt_oss_20b
widget_predictor_llm = gpt_oss_20b
credibility_llm = gpt_oss_20b
generate_cover_llm = gpt_oss_20b
# Structured extraction only (title/instruction/schedule) — low reasoning
# effort is enough and keeps the interactive create-flow snappy.
research_schedule_llm = gpt_oss_120b_low
