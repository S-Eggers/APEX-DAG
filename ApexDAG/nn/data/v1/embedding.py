import logging
from enum import Enum
from pathlib import Path

import torch
from dotenv import load_dotenv

from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False

_model_cache: dict[str, any] = {}

class EmbeddingType(Enum):
    FASTTEXT = 1
    GEMINI_STANDARD = 2
    GEMINI_CODE = 3
    GRAPHCODEBERT = 4
    CODEBERT = 5
    BERT = 6

class Embedding:
    def __init__(self, type: EmbeddingType, max_output_dim: int = 768):
        self.type = type
        self._model = None
        self._embedding_model_name = ""
        self._max_output_dim = max_output_dim
        self._initialize_model()

    @property
    def dimension(self) -> int:
        """Returns the deterministic tensor dimension for the selected embedding type."""
        if self.type == EmbeddingType.FASTTEXT:
            return 300
        return self._max_output_dim

    def _initialize_model(self):
        load_dotenv()
        match self.type:
            case EmbeddingType.FASTTEXT:
                self._embedding_model_name = "cc.en.300.bin"
                if self._embedding_model_name not in _model_cache:
                    logger.info(f"Loading FastText model: {self._embedding_model_name}...")
                    import fasttext

                    package_root = Path(__file__).parent.absolute()
                    model_path = package_root / self._embedding_model_name
                    if not model_path.exists():
                        model_path = Path(self._embedding_model_name)

                    try:
                        _model_cache[self._embedding_model_name] = fasttext.load_model(str(model_path))
                        logger.info("FastText model loaded successfully.")
                    except Exception as e:
                        logger.error(f"Failed to load FastText model at {model_path}: {e}")
                        raise
                self._model = _model_cache[self._embedding_model_name]

            case EmbeddingType.GEMINI_STANDARD | EmbeddingType.GEMINI_CODE:
                self._embedding_model_name = "gemini-embedding-001"
                client = genai.Client()
                self._model = client.models

            case _:
                logger.warning(f"Embedding initialization for {self.type} is not implemented.")

    def _get_sentence_vector_fast_text(self, sequence: str) -> torch.Tensor:
        vector = self._model.get_sentence_vector(sequence)
        return torch.tensor(vector, dtype=torch.float32)

    def _get_sentence_vector_gemini(
        self, sequence: str, config: types.EmbedContentConfig | None = None
    ) -> torch.Tensor:
        result = self._model.embed_content(
            model=self._embedding_model_name, contents=sequence, config=config
        )
        return torch.tensor(result.embeddings[0].values, dtype=torch.float32)

    def embed(self, sequence: str) -> torch.Tensor:
        if "\n" in sequence:
            sequence = sequence.replace("\n", " ")

        match self.type:
            case EmbeddingType.FASTTEXT:
                return self._get_sentence_vector_fast_text(sequence)
            case EmbeddingType.GEMINI_STANDARD:
                config = types.EmbedContentConfig(
                    task_type="CLASSIFICATION",
                    output_dimensionality=self._max_output_dim,
                )
                return self._get_sentence_vector_gemini(sequence, config)
            case EmbeddingType.GEMINI_CODE:
                config = types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=self._max_output_dim,
                )
                return self._get_sentence_vector_gemini(sequence, config)
            case _:
                raise NotImplementedError(f"Embedding type {self.type} not implemented")
