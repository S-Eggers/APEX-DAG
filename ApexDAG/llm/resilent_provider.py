import logging
import random
import time
from typing import TypeVar

from pydantic import BaseModel

from ApexDAG.util.logger import configure_apexdag_logger

from .llm_provider import LLMResponse, StructuredLLMProvider

configure_apexdag_logger()
logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class ResilientProvider(StructuredLLMProvider):
    """
    A decorator that adds exponential backoff to any StructuredLLMProvider.
    """

    def __init__(self, inner: StructuredLLMProvider, max_retries: int = 5) -> None:
        self.inner = inner
        self.max_retries = max_retries

    def generate(self, prompt: str, response_schema: type[T], **kwargs: dict) -> LLMResponse[T]:
        initial_delay = 30.0
        factor = 2.0

        for attempt in range(self.max_retries):
            try:
                return self.inner.generate(prompt, response_schema, **kwargs)
            except Exception as e:
                err_msg = str(e).lower()
                # Categorize retryable errors
                is_retryable = any(x in err_msg for x in ["429", "quota", "exhausted", "503", "busy", "limit", "timeout"])

                if not is_retryable or attempt == self.max_retries - 1:
                    raise e

                delay = initial_delay * (factor**attempt)
                jittered_delay = random.uniform(1.0, delay)

                logger.warning(f"Retrying LLM call in {jittered_delay:.2f}s due to: {e}")
                time.sleep(jittered_delay)

        raise RuntimeError("Max retries exceeded")
