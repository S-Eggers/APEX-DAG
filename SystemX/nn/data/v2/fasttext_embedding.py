import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_FASTTEXT_CACHE: dict[str, object] = {}

_VECTOR_CACHE: dict[tuple[str, str], torch.Tensor] = {}
_VECTOR_CACHE_MAX = 1_000_000
_ZERO_VECTOR = torch.zeros(300, dtype=torch.float32)

_V1_PATH = Path(__file__).parent.parent / "v1" / "cc.en.300.bin"
_REPO_ROOT_PATH = Path(__file__).parents[4] / "cc.en.300.bin"
_DEFAULT_MODEL_PATH = _V1_PATH if _V1_PATH.exists() else _REPO_ROOT_PATH

class FastTextEmbeddingV2:
    """Lightweight 300-d FastText embedding for V2 compute-hub classification."""

    dimension: int = 300

    def __init__(self, model_path: Path | None = None) -> None:
        path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        cache_key = str(path)

        if cache_key not in _FASTTEXT_CACHE:
            import fasttext

            logger.info("Loading FastText model from %s ...", path)
            if not path.exists():
                raise FileNotFoundError(f"FastText model not found at {path}. Download cc.en.300.bin and place it in SystemX/nn/data/v1/.")
            _FASTTEXT_CACHE[cache_key] = fasttext.load_model(str(path))
            logger.info("FastText model loaded.")

        self._model = _FASTTEXT_CACHE[cache_key]
        self._cache_key = cache_key

    def embed(self, code_snippet: str) -> torch.Tensor:
        if not code_snippet or str(code_snippet).strip() in ("", "None"):
            return _ZERO_VECTOR
        clean = str(code_snippet).replace("\n", " ").strip()
        key = (self._cache_key, clean)
        cached = _VECTOR_CACHE.get(key)
        if cached is not None:
            return cached
        vector = self._model.get_sentence_vector(clean)
        tensor = torch.tensor(vector, dtype=torch.float32)
        if len(_VECTOR_CACHE) < _VECTOR_CACHE_MAX:
            _VECTOR_CACHE[key] = tensor
        return tensor
