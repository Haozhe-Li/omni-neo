from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model

gpt_oss_120b_low = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="low")
gpt_oss_120b_high = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="high")
gpt_oss_20b = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1)
qwen_3_6_27b = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.2)
gemini_flash_lite_latest = init_chat_model("google_genai:gemini-flash-lite-latest")
gemini_flash = init_chat_model("google_genai:gemini-3-flash-preview")
llama3_1_8b = ChatGroq(model="llama-3.1-8b-instant")
glm_4_7 = ChatCerebras(model="zai-glm-4.7", temperature=0.2)
gpt_oss_120b_medium = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="medium")
gemma_4_31b = ChatCerebras(model="gemma-4-31b", temperature=0.2)

fast_llm = gpt_oss_120b_low
pro_llm = gemma_4_31b
