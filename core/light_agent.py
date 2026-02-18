from langchain.chat_models import init_chat_model
from core.tools.web_search import tavily_search
from langgraph.prebuilt import create_react_agent
from core.database.postgresql_saver import checkpointer

model = init_chat_model("groq:openai/gpt-oss-20b")

omni_light_agent = create_react_agent(
    model=model,
    tools=[tavily_search],
    prompt=(
        """
        You are a agent called Omni Light. You receive a user query and deliver a quick and warm response.

        Tools:
        - tavily_search: search the web for relevant information.

        Rules:
        - Use tavily_search if and only if the user's query requires time-sensitive information.
        - Otherwise, answer directly yourself.
        """
    ),
    name="light_agent",
    checkpointer=checkpointer,
)
