from pydantic import BaseModel, Field


class Memories(BaseModel):
    user_profile: str | None = None
    current_focus: str | None = None
    interaction_style: str | None = None
    avoid_topics: str | None = None


class UpdateMemoriesRequest(BaseModel):
    past_queries: list[str]
    past_memories: Memories | None = None


class Personalization(BaseModel):
    response_language: str = "Follow User's Query Language"
    memories: Memories | None = None
    user_local_datetime: str | None = None
    user_location: str | None = None


class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None
    follow_up_content: str | None = None
    personalization: Personalization | None = None


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


class SupervisorOutput(BaseModel):
    answer: str = Field(description="The markdown report.")
    sources: list[dict] = Field(
        description="The final sources used to generate the answer. Leave a empty list if no sources are used."
    )
    assets: list[str] = Field(
        description="A list of URL of images or other assets used to generate the answer. Leave a empty list if no assets are used."
    )


class CodeExpertOutput(BaseModel):
    code: str = Field(
        description="The Python Code (If you only draw a graph, you can leave this empty)"
    )
    code_output: str = Field(
        description="The output of the Python Code (If any, leave empty if no output)"
    )
    assets: list[str] = Field(
        description="A list of URLs of images you draw. Leave a empty list if no assets are used."
    )


class StockExpertOutput(BaseModel):
    report: str = Field(description="The analysis report of the stock.")
    assets: list[str] = Field(
        description="A list of URLs of images you draw. Leave a empty list if no assets are used."
    )
