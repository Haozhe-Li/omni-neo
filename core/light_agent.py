from langchain.chat_models import init_chat_model
from core.utils.light_tools import (
    google_search_light,
    google_search_places_light,
    get_stock_data_light,
    get_weather_light,
)
from langchain.agents import create_agent
from langchain.agents import AgentState
from core.database.postgresql_saver import checkpointer
from langchain.agents.middleware import (
    ToolRetryMiddleware,
    ToolCallLimitMiddleware,
    ModelCallLimitMiddleware,
)
from typing import Any


class LightAgentState(AgentState):
    sources: list[dict[str, str]]
    map: list[dict[str, Any]]
    stock: dict[str, Any]
    weather: dict[str, Any]


model = init_chat_model("groq:openai/gpt-oss-20b")


LIGHT_AGENT_SYSTEM_PROMPT = """
        You are a agent called Omni Light. You will provide answers to user.

        Tools:
        - google_search_light: search the web for relevant information.
        - google_search_places_light: search places (restaurants, cafes, stores, POIs) for relevant local recommendations.
        - get_stock_data_light: get latest stock data by ticker symbol.
        - get_weather_light: get current weather for a location.

        Rules:
        - Use google_search_light if and only if the user's query requires time-sensitive information.
        - Use google_search_places_light when user asks for places or local recommendations. Important: you MUST explicitly includes location in query, e.g. coffee shop in navy pier.
        - Use get_stock_data_light when user asks stock price/metrics for a specific ticker.
        - Use get_weather_light when user asks about current weather in a specific location. If user does not provide a location, use the personalization info gave to you.
        - Otherwise, answer directly yourself.
        - You might be given some user personalization information. Use it if it is helpful to answer the query.

        Answer:
        - Use markdown format only.
        - DO NOT USE any headings in markdown, including #, ##, etc.
        - Keep your answer clear, warm and casual.
        - Don't be Perfunctory, try to respond at least 3 sentences if you can. If the answer is short, you can provide some additional relevant information to the user.

        Output:
        - Return plain text markdown answer only.
    """

omni_light_agent = create_agent(
    model=model,
    tools=[
        google_search_light,
        google_search_places_light,
        get_stock_data_light,
        get_weather_light,
    ],
    system_prompt=LIGHT_AGENT_SYSTEM_PROMPT,
    name="Omni Light",
    checkpointer=checkpointer,
    middleware=[
        ToolRetryMiddleware(
            max_retries=2,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
        ToolCallLimitMiddleware(run_limit=2),
        ModelCallLimitMiddleware(run_limit=5),
    ],
    state_schema=LightAgentState,
)
