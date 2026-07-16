from typing import Protocol

from pydantic import BaseModel


class OperationMetric(BaseModel):
    name: str
    count: int


class PromptTemplate(Protocol):
    """Protocol for all prompt templates to ensure consistency."""

    def render_user_message(self, **kwargs: object) -> str: ...

    def render_system_message(self, **kwargs: object) -> str: ...
