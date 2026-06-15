from langchain_cerebras import ChatCerebras
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model

gpt_oss_120b_low = ChatGroq(model="openai/gpt-oss-120b", temperature=0.2, reasoning_effort="low")
gpt_oss_120b_high = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1, reasoning_effort="low")
gemini_flash_lite_latest = init_chat_model("google_genai:gemini-flash-lite-latest")
gemini_flash_latest = init_chat_model("google_genai:gemini-flash-latest")
llama3_1_8b = ChatCerebras(model="llama3.1-8b")
glm_4_7 = ChatCerebras(model="zai-glm-4.7", temperature=0.1)