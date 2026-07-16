import os

from SystemX.llm.local_openai_provider import LocalOpenAIProvider

def _resolve_mlx_base_url() -> str:
    """Resolve the MLX server's OpenAI-compatible base URL from the environment."""
    url = os.getenv("MLX_SERVER")
    if not url or not url.strip():
        raise ValueError(
            "MLX_SERVER environment variable is missing. Set it in your .env to the "
            "MLX server's base URL, e.g. MLX_SERVER=http://10.0.0.2:8080"
        )
    url = url.strip().rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"
    return url

class MLXProvider(LocalOpenAIProvider):
    """Structured-output provider for a remote MLX server (mlx_lm.server)."""

    def __init__(self, model_name: str, api_key: str | None = None, max_output_tokens: int | None = None) -> None:
        super().__init__(
            model_name=model_name,
            base_url=_resolve_mlx_base_url(),
            api_key=api_key or os.getenv("MLX_SERVER_API_KEY"),
            max_output_tokens=max_output_tokens,
        )
