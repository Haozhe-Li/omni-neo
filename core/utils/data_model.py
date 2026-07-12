from typing import Literal

from pydantic import BaseModel, Field


class Personalization(BaseModel):
    response_language: str = "Follow User's Query Language"
    memory_enabled: bool = False
    user_local_datetime: str | None = None
    user_location: str | None = None
    user_unit: str | None = None


class QueryRequest(BaseModel):
    query: str
    thread_id: str | None = None
    follow_up_content: str | None = None
    personalization: Personalization | None = None
    attached_file_ids: list[dict[str, str]] | None = None
    mode: Literal["fast", "pro"] = "fast"
    skill: str | None = None
    # 1-indexed turn number for this exchange, assigned by the frontend
    # (the backend has no other source of truth for turn ordering). Every
    # citation produced during this turn is stamped with it, so
    # `/check_source` can later confine a claim to sources that existed by
    # its turn — see `core/utils/citations.py`.
    turn: int | None = None


class AutoCompleteRequest(BaseModel):
    text: str


class GenerateCoverRequest(BaseModel):
    title: str


class CheckSourceRequest(BaseModel):
    thread_id: str
    text_selection: str
    # The turn the checked claim belongs to. Only sources introduced at or
    # before this turn are eligible matches — None disables the cutoff.
    turn: int | None = None


class Decision(BaseModel):
    is_smart: bool = Field(
        description="True if query requires deep thinking, coding, or research. False if simple."
    )


class LightAgentOutput(BaseModel):
    answer: str = Field(description="The final answer to the user's query.")
    use_search: bool = Field(description="Whether you have used google_search.")


class ResearchHelperOutput(BaseModel):
    response: str = Field(description="The response to the user's query.")
    ready_to_begin_research: bool = Field(
        description="Whether you are ready to begin research."
    )
    rewritten_query: str = Field(description="The rewritten query for research.")
    questions_for_user: list[dict] = Field(
        description="Questions for the user to clarify the query."
    )


class GuidedLearningQuestion(BaseModel):
    question: str = Field(description="The question to ask the user.")
    options: list[str] = Field(description="The options for the question.")
    answer: int | None = Field(
        default=None,
        description="The 0-indexed correct option if this is a quiz. Leave as null if it's just a regular question.",
    )


class GuidedLearningFlashcard(BaseModel):
    key: str = Field(description="The key or terminology for the flashcard.")
    value: str = Field(description="The detailed definition or explanation.")


class GuidedLearningOutput(BaseModel):
    response: str = Field(description="The response to the user's query.")
    questions_for_user: list[GuidedLearningQuestion] = Field(
        description="Questions for the user to test their knowledge, or quizzes etc."
    )
    flashcard: list[GuidedLearningFlashcard] = Field(
        description="Flashcards for the user to learn the topic."
    )
    note: str = Field(
        description="The detailed markdown note covering the topic when user needs a large amount of notes."
    )


class SupervisorOutput(BaseModel):
    title: str = Field(
        description="A very short title of the report, no more than 5 words."
    )
    answer: str = Field(description="The markdown report.")


class CodeExpertOutput(BaseModel):
    code: str = Field(
        description="The Python Code (If you only draw a graph, you can leave this empty)"
    )
    code_output: str = Field(
        description="The output of the Python Code (If any, leave empty if no output)"
    )
    assets: list[str] = Field(
        description="A list of URLs of images you draw. You MUST INCLUDE ALL IMAGE URL YOU GENERATED in this field. Leave a empty list if no images are created."
    )


class StockExpertOutput(BaseModel):
    report: str = Field(description="The analysis report of the stock.")
    assets: list[str] = Field(
        description="A list of URLs of images you draw. You MUST INCLUDE ALL IMAGE URL YOU GENERATED in this field. Leave a empty list if no images are created."
    )
