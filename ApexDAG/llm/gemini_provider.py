import os

from google import genai
from google.genai import types

from .llm_provider import LLMResponse, T


class GeminiProvider:
    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is missing.")
        self.client = genai.Client(api_key=key)
        self.model_name = model_name

    def generate(self, prompt: str, response_schema: type[T], system_instruction: str | None = None, temperature: float = 0.1) -> LLMResponse[T]:
        config = types.GenerateContentConfig(response_mime_type="application/json", response_schema=response_schema, temperature=temperature, system_instruction=system_instruction)
        response = self.client.models.generate_content(model=self.model_name, contents=prompt, config=config)

        tokens = getattr(response.usage_metadata, "total_token_count", 0)
        data = response_schema.model_validate_json(response.text)
        return LLMResponse(data=data, token_usage=tokens)
