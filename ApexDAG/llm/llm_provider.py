from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMResponse(Generic[T]):
    """Standardized wrapper for LLM outputs and metadata."""

    def __init__(self, data: T, token_usage: int) -> None:
        self.data = data
        self.token_usage = token_usage


class StructuredLLMProvider(Protocol):
    """Protocol for providers that return Pydantic-validated data."""

    def generate(self, prompt: str, response_schema: type[T], system_instruction: str | None = None, temperature: float = 0.1) -> LLMResponse[T]: ...
