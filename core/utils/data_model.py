from pydantic import BaseModel, Field


class Personalization(BaseModel):
    response_language: str = "Follow User's Query Language"
    memories: list[str] = []
    user_local_datetime: str = None
    user_location: str = None


class QueryRequest(BaseModel):
    query: str
    thread_id: str = None
    follow_up_content: str = None
    personalization: Personalization = Personalization()


class CheckSourceRequest(BaseModel):
    source: dict  # Check if source is a dict
    text_selection: str


class Decision(BaseModel):
    is_smart: bool = Field(
        description="True if query requires deep thinking, coding, or research. False if simple."
    )


class LightAgentOutput(BaseModel):
    answer: str = Field(description="The final answer to the user's query.")
    use_search: bool = Field(description="Whether you have used tavily_search.")
