import os
import torch
import logging
from enum import Enum
from logging import Logger
from dotenv import load_dotenv
from typing import Optional, Dict

try:
    from google import genai
    from google.genai import types

    GOOGLE_AI_AVAILABLE = True
except ImportError:
    print(
        "WARNING: 'google-generativeai' library not found. Gemini tests will be skipped."
    )
    print("Install it with: pip install google-generativeai")
    GOOGLE_AI_AVAILABLE = False

_model_cache: Dict[str, any] = {}


class EmbeddingType(Enum):
    FASTTEXT = 1
    GEMINI_STANDARD = 2
    GEMINI_CODE = 3
    GRAPHCODEBERT = 4
    CODEBERT = 5
    BERT = 6


class Embedding:
    def __init__(self, type: EmbeddingType, logger: Logger, max_output_dim=768):
        self.type = type
        self.logger = logger
        self._model = None
        self._embedding_model_name = ""
        self._max_output_dim = max_output_dim
        self._initialize_model()

    def _initialize_model(self):
        match self.type:
            case EmbeddingType.FASTTEXT:
                self._embedding_model_name = "cc.en.300.bin"
                if self._embedding_model_name not in _model_cache:
                    self.logger.info(
                        f"Loading FastText model: {self._embedding_model_name}..."
                    )
                    import fasttext

                    _model_cache[self._embedding_model_name] = fasttext.load_model(
                        self._embedding_model_name
                    )
                    self.logger.info("FastText model loaded.")
                self._model = _model_cache[self._embedding_model_name]

            case EmbeddingType.GEMINI_STANDARD | EmbeddingType.GEMINI_CODE:
                self._embedding_model_name = "gemini-embedding-001"
                client = genai.Client()
                self._model = client.models

            case EmbeddingType.GRAPHCODEBERT | EmbeddingType.CODEBERT:
                # Not implemented
                pass
            case _:
                # Default case
                pass

    def _get_sentence_vector_fast_text(self, sequence: str) -> torch.Tensor:
        vector = self._model.get_sentence_vector(sequence)
        return torch.tensor(vector, dtype=torch.float32)

    def _get_sentence_vector_gemini(
        self, sequence: str, config: Optional[types.EmbedContentConfig] = None
    ) -> torch.Tensor:
        result = self._model.embed_content(
            model=self._embedding_model_name, contents=sequence, config=config
        )
        return torch.tensor(result.embeddings[0].values, dtype=torch.float32)

    def embed(self, sequence: str) -> torch.Tensor:
        if "\n" in sequence:
            self.logger.warning(
                "Sequence contains newline characters, which will be replaced by spaces."
            )
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


if __name__ == "__main__":
    # --- Example Usage ---
    import sys
    import traceback

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    def print_embedding_result(
        model_name: str, text: str, tensor: Optional[torch.Tensor]
    ):
        print("-" * 50)
        print(f"Model: {model_name}")
        print(f"Input: '{text[:80]}...'")  # Print truncated input
        if tensor is not None:
            print(f"Output Shape: {tensor.shape}")
            print(f"Output dtype: {tensor.dtype}")
            print(f"Output Vector (first 5 dims): {tensor[:5]}")
        else:
            print("Output: FAILED")
        print("-" * 50 + "\n")

    logger.info("--- STARTING FASTTEXT DEMO ---")
    try:
        fasttext_embedder = Embedding(EmbeddingType.FASTTEXT, logger)
        fasttext_inputs = [
            "This is a simple sentence for testing.",
            "How does the embedding model handle questions?",
            "Processing text with numbers like 1, 2, and 3.",
            "A sentence with a newline character\nwhich should be replaced.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        for text in fasttext_inputs:
            embedding_tensor = fasttext_embedder.embed(text)
            print_embedding_result("FastText", text, embedding_tensor)
    except (FileNotFoundError, ImportError) as e:
        logger.error(f"Could not run FastText demo: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during the FastText demo: {e}")

    if GOOGLE_AI_AVAILABLE:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning(
                "GEMINI_API_KEY not found in .env file or environment variables. Skipping Gemini tests."
            )
        else:
            try:
                logger.info("--- STARTING GEMINI STANDARD DEMO ---")
                gemini_standard_embedder = Embedding(
                    EmbeddingType.GEMINI_STANDARD, logger
                )
                gemini_standard_inputs = [
                    "The movie was fantastic, a true masterpiece of cinema.",
                    "This product is poorly made and broke after one use.",
                    "Review of a restaurant: The service was slow but the food was delicious.",
                    "Is this email spam or important?",
                    "A neutral statement about the weather today.",
                ]
                for text in gemini_standard_inputs:
                    embedding_tensor = gemini_standard_embedder.embed(text)
                    print_embedding_result("Gemini Standard", text, embedding_tensor)

                logger.info("--- STARTING GEMINI CODE DEMO ---")
                gemini_code_embedder = Embedding(EmbeddingType.GEMINI_CODE, logger)
                gemini_code_inputs = [
                    "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)",
                    "How to sort a dictionary by value in Python?",
                    "SELECT customer_name, order_total FROM orders WHERE order_date > '2023-01-01'",
                    "Explain the difference between a list and a tuple.",
                    "// JavaScript function to find the max value in an array\nfunction findMax(arr) { return Math.max(...arr); }",
                ]
                for text in gemini_code_inputs:
                    embedding_tensor = gemini_code_embedder.embed(text)
                    print_embedding_result("Gemini Code", text, embedding_tensor)

            except Exception as e:
                logger.error(f"An error occurred during Gemini API calls: {e}")
                traceback.print_exc(file=sys.stdout)
