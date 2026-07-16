import os
from typing import TypeVar

from groq import Groq
from pydantic import BaseModel

from .llm_provider import LLMResponse

T = TypeVar("T", bound=BaseModel)


class GroqProvider:
    """Groq implementation of the StructuredLLMProvider."""

    def __init__(self, model_name: str, api_key: str | None = None) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")

        self.client = Groq(api_key=key)
        self.model_name = model_name

    def generate(self, prompt: str, response_schema: type[T], system_instruction: str | None = None, temperature: float = 0.1) -> LLMResponse[T]:

        schema_instructions = f"\nReturn the output as JSON matching this schema: {response_schema.model_json_schema()}"

        full_system_instruction = (system_instruction or "") + schema_instructions

        messages = [{"role": "system", "content": full_system_instruction}, {"role": "user", "content": prompt}]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq returned an empty response.")

        data = response_schema.model_validate_json(content)
        tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0

        return LLMResponse(data=data, token_usage=tokens)
