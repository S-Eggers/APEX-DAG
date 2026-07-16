import os
import re

from google import genai
from google.genai import types

from .llm_provider import LLMResponse, ProviderRateLimitError, T

def _extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply."""
    if not text:
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text

class GeminiProvider:
    def __init__(self, model_name: str, api_key: str | None = None, max_output_tokens: int | None = None, thinking_budget: int = 0) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is missing.")
        self.client = genai.Client(api_key=key)
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget

    def generate(self, prompt: str, response_schema: type[T], system_instruction: str | None = None, temperature: float = 0.1) -> LLMResponse[T]:
        config_kwargs: dict = dict(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=temperature,
            system_instruction=system_instruction,
            max_output_tokens=self.max_output_tokens,
        )
        if self.model_name.lower().startswith("gemma"):
            if self.thinking_budget and self.thinking_budget > 0:
                config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level="high")
        else:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            response = self.client.models.generate_content(model=self.model_name, contents=prompt, config=config)
        except Exception as e:
            self._handle_provider_error(e)

        tokens = getattr(response.usage_metadata, "total_token_count", 0)
        if not response.text:
            raise ValueError(
                f"{self.model_name} returned no text (usage: {tokens} tokens) - "
                "likely max_output_tokens too small for the thinking budget."
            )
        data = response_schema.model_validate_json(_extract_json(response.text))
        return LLMResponse(data=data, token_usage=tokens)

    def _handle_provider_error(self, error: Exception) -> None:
        """Parses Google-specific errors into standard domain exceptions."""
        error_str = str(error)

        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            match = re.search(r"'retryDelay':\s*'([0-9.]+)s'", error_str)
            if match:
                delay = float(match.group(1))
                raise ProviderRateLimitError(retry_after=delay, original_error=error)

            raise ProviderRateLimitError(retry_after=60.0, original_error=error)

        raise error
