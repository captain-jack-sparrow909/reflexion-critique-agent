from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")



class AnswerQuestion(BaseModel):
    """Answer question"""
    answer: str = Field(description="~250 words detailed answer to the question")
    reflection: Reflection = Field(description="your reflection on the initial answer")
    search_queries: list[str] = Field(description="1-3 search queries for researching improvements to address the critique of your current answer.")


class ReviseAnswer(AnswerQuestion):
    """Revise your original answer to your question"""
    references: list[str] = Field(description="Citations motivating your current answer")
    