from langchain.chat_models import init_chat_model
from core.tools.web_search import tavily_search
from langchain.agents import create_agent
from core.database.postgresql_saver import checkpointer
from langchain.agents.middleware import ToolRetryMiddleware, ToolCallLimitMiddleware
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from pydantic import BaseModel, Field


class LightAgentOutput(BaseModel):
    answer: str = Field(description="The final answer to the user's query.")
    use_search: bool = Field(description="Whether you have used tavily_search.")


model = init_chat_model("openai:gpt-4.1-nano-2025-04-14")

omni_light_agent = create_agent(
    model=model,
    tools=[tavily_search],
    system_prompt="""
        You are a agent called Omni Light. You receive a user query and deliver a quick and warm response.

        Tools:
        - tavily_search: search the web for relevant information.

        Rules:
        - Use tavily_search if and only if the user's query requires time-sensitive information.
        - Otherwise, answer directly yourself.
        - The model behind you is gpt-4.1-nano. Don't reveal this unless the user explicitly asks.

        Answer:
        - Use markdown format only.
        - DO NOT USE any headings in markdown, including #, ##, etc. 
        - Keep your answer simple, warm and casual.

        Output:
        Output to pydantic model:
        - answer: The final answer to the user's query.
        - use_search: Whether you have used tavily_search.
    """,
    name="light_agent",
    checkpointer=checkpointer,
    middleware=[
        ToolRetryMiddleware(
            max_retries=2,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
        ToolCallLimitMiddleware(run_limit=2),
    ],
    response_format=ProviderStrategy(LightAgentOutput),
)
