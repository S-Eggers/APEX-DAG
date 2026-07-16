import logging
import random
import time
from typing import TypeVar

from pydantic import BaseModel

from .llm_provider import LLMResponse, ProviderRateLimitError, StructuredLLMProvider

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

class ResilientProvider(StructuredLLMProvider):
    """A decorator that adds exponential backoff and respects explicit rate limits."""

    def __init__(self, inner: StructuredLLMProvider, max_retries: int = 5) -> None:
        self.inner = inner
        self.max_retries = max_retries

    def generate(self, prompt: str, response_schema: type[T], system_instruction: str | None = None, temperature: float = 0.1) -> LLMResponse[T]:

        initial_delay = 30.0
        factor = 2.0

        for attempt in range(self.max_retries):
            try:
                return self.inner.generate(prompt=prompt, response_schema=response_schema, system_instruction=system_instruction, temperature=temperature)

            except ProviderRateLimitError as limit_err:
                if attempt == self.max_retries - 1:
                    logger.error("Max retries reached while handling explicit rate limits.")
                    raise limit_err.original_error from limit_err

                exact_delay = limit_err.retry_after + 1.0
                logger.warning(f"Provider requested explicit backoff. Sleeping for {exact_delay:.2f}s.")
                time.sleep(exact_delay)

            except Exception as e:
                err_msg = str(e).lower()
                is_retryable = any(x in err_msg for x in ["quota", "exhausted", "503", "busy", "limit", "timeout"])

                if not is_retryable or attempt == self.max_retries - 1:
                    logger.error(f"Provider failed irreversibly or max retries reached: {e}")
                    raise

                delay = initial_delay * (factor**attempt)
                jittered_delay = random.uniform(1.0, delay)

                logger.warning(f"Retrying LLM call in {jittered_delay:.2f}s due to generic error: {e}")
                time.sleep(jittered_delay)

        raise RuntimeError("Max retries exceeded")
