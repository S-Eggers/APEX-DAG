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

class ProviderRateLimitError(Exception):
    """Raised when an LLM provider explicitly dictates a backoff period."""

    def __init__(self, retry_after: float, original_error: Exception) -> None:
        self.retry_after = retry_after
        self.original_error = original_error
        super().__init__(f"Provider rate limit exhausted. Must wait {retry_after}s.")

