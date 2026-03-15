from langchain_cerebras import ChatCerebras
from langchain.chat_models import init_chat_model

gpt_oss_120b_low = ChatCerebras(model="gpt-oss-120b", reasoning_effort="low")
gemini_flash_lite_latest = init_chat_model("google_genai:gemini-flash-lite-latest")
gemini_flash_latest = init_chat_model("google_genai:gemini-flash-latest")
llama3_1_8b = ChatCerebras(model="llama3.1-8b")
