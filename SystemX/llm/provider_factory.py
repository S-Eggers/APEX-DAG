from SystemX.llm.config import Config
from SystemX.llm.gemini_provider import GeminiProvider
from SystemX.llm.groq_provider import GroqProvider
from SystemX.llm.llm_provider import StructuredLLMProvider
from SystemX.llm.local_openai_provider import LocalOpenAIProvider
from SystemX.llm.mlx_provider import MLXProvider

_PROVIDER_REGISTRY: dict[str, type] = {
    "google": GeminiProvider,
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "mlx": MLXProvider,
    "local": LocalOpenAIProvider,
    "openai_compatible": LocalOpenAIProvider,
}

class ProviderFactory:
    @staticmethod
    def create(config: Config) -> StructuredLLMProvider:
        provider_type = config.llm_provider.lower().strip()
        provider_cls = _PROVIDER_REGISTRY.get(provider_type)
        if provider_cls is None:
            raise ValueError(f"Unsupported llm_provider: '{config.llm_provider}'. Must be one of {sorted(_PROVIDER_REGISTRY)}")
        if provider_cls is GeminiProvider:
            return GeminiProvider(
                model_name=config.model_name,
                max_output_tokens=config.max_output_tokens,
                thinking_budget=config.thinking_budget,
            )
        if provider_cls is MLXProvider:
            return MLXProvider(
                model_name=config.model_name,
                max_output_tokens=config.max_output_tokens,
            )
        if provider_cls is LocalOpenAIProvider:
            return LocalOpenAIProvider(
                model_name=config.model_name,
                base_url=config.base_url,
                max_output_tokens=config.max_output_tokens,
            )
        return provider_cls(model_name=config.model_name)
