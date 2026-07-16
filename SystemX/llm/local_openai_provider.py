import json
import os
from typing import TypeVar

import httpx
from openai import OpenAI
from pydantic import BaseModel

from .llm_provider import LLMResponse

T = TypeVar("T", bound=BaseModel)

def _example_from_json_schema(schema: dict, defs: dict | None = None) -> object:
    """Build a compact *example instance* from a JSON schema."""
    defs = defs if defs is not None else schema.get("$defs", {})
    if "$ref" in schema:
        return _example_from_json_schema(defs.get(schema["$ref"].split("/")[-1], {}), defs)
    if "enum" in schema:
        return "<one of: " + " | ".join(str(v) for v in schema["enum"]) + ">"
    if "anyOf" in schema:
        return _example_from_json_schema(schema["anyOf"][0], defs)
    t = schema.get("type")
    if t == "object":
        return {k: _example_from_json_schema(v, defs) for k, v in schema.get("properties", {}).items()}
    if t == "array":
        return [_example_from_json_schema(schema.get("items", {}), defs)]
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return False
    return "<text>"

def _extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply."""
    if not text:
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text

class LocalOpenAIProvider:
    """Structured-output provider for any OpenAI-compatible **local** server."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_output_tokens: int | None = None,
    ) -> None:
        key = api_key or os.getenv("LOCAL_LLM_API_KEY") or "not-needed"
        self.client = OpenAI(
            base_url=base_url,
            api_key=key,
            timeout=httpx.Timeout(300.0, connect=10.0),
            max_retries=3,
        )
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens

    def generate(self, prompt: str, response_schema: type[T], system_instruction: str | None = None, temperature: float = 0.1) -> LLMResponse[T]:
        example = _example_from_json_schema(response_schema.model_json_schema())
        schema_instructions = (
            "\n\nRespond with ONLY a single JSON object of exactly this shape, replacing the "
            "placeholder values. Output nothing else - no schema, no explanation, no code fence:\n"
            f"{json.dumps(example)}"
        )
        full_system_instruction = (system_instruction or "") + schema_instructions

        instruction = full_system_instruction.strip()
        user_content = f"{instruction}\n\n{prompt}" if instruction else prompt
        messages = [
            {"role": "user", "content": user_content},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=self.max_output_tokens,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError(f"{self.model_name} (local) returned an empty response.")

        data = response_schema.model_validate_json(_extract_json(content))
        tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0

        return LLMResponse(data=data, token_usage=tokens)
