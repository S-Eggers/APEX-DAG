import logging
from typing import Protocol, runtime_checkable

import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

@runtime_checkable
class EmbeddingProtocol(Protocol):
    """Shared interface for all V2 embedding backends."""

    dimension: int

    def embed(self, code_snippet: str) -> torch.Tensor: ...

class TransformerEmbedding:
    """Transformer-based embedding for compute-hub code snippets."""

    def __init__(self, model_name: str = "microsoft/graphcodebert-base") -> None:
        self.model_name = model_name
        self._max_output_dim = 768

        logger.info("Loading HF Tokenizer and Model: %s", self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("Transformer model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load HuggingFace model %s: %s", self.model_name, e)
            raise

    @property
    def dimension(self) -> int:
        return self._max_output_dim

    @torch.no_grad()
    def embed(self, code_snippet: str) -> torch.Tensor:
        if not code_snippet or str(code_snippet).strip() == "None":
            return torch.zeros(self.dimension, dtype=torch.float32)

        clean_code = str(code_snippet).replace("\n", " ").strip()
        inputs = self.tokenizer(
            clean_code,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.device)

        outputs = self.model(**inputs)
        return outputs.last_hidden_state[0, 0, :].cpu()

CodeBERTEmbedding = TransformerEmbedding
