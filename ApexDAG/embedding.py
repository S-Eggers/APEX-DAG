import torch
from enum import Enum
from logging import Logger


class EmbeddingType(Enum):
    FASTTEXT = 1
    GEMINI_STANDARD = 2
    GEMINI_CODE = 3
    GRAPHCODEBERT = 4
    CODEBERT = 5
    BERT = 6

class Embedding:
    def __init__(self, type: EmbeddingType, logger: Logger):
        self.type = type
        self.logger = logger
        match self.type:
            case EmbeddingType.FASTTEXT:
                import fasttext
                self._model = fasttext.load_model("cc.en.300.bin")
                self._embedding_model = "cc.en.300.bin"
            case EmbeddingType.GEMINI_STANDARD | EmbeddingType.GEMINI_CODE:
                from google import genai
                from genai import types
                self._model = client = genai.Client()
                self._embedding_model = "gemini-embedding-001"
            case EmbeddingType.GRAPHCODEBERT:
                pass
            case EmbeddingType.CODEBERT:
                pass
            case _:
                pass
    
    def _get_sentence_vector_fast_text(self, sentence: str) -> torch.Tensor:
        return torch.tensor(self._model.get_sentence_vector(sentence), dtype=torch.float32)

    def embed(self, sequence: str) -> torch.Tensor:
        if "\n" in sentence:
            self.logger(f"ERROR: Sentence contains newline character: {sentence}")
            sentence = sentence.replace("\n", " ")
        
        match self.type:
            case EmbeddingType.FASTTEXT:
                return self._get_sentence_vector_fast_text(sequence)
            case EmbeddingType.GEMINI_STANDARD:
                result = self._model.models.embed_content(
                    model=self._embedding_model,
                    contents=sequence,
                    config=types.EmbedContentConfig(task_type="CLASSIFICATION")
                )
                return torch.tensor(result.embeddings, dtype=torch.float32)
            case EmbeddingType.GEMINI_CODE:
                result = self._model.models.embed_content(
                    model=self._embedding_model,
                    contents=sequence,
                    config=types.EmbedContentConfig(task_type="CODE_RETRIEVAL_QUERY")
                )
                return torch.tensor(result.embeddings, dtype=torch.float32)
            case _:
                raise NotImplementedError(f"Embedding type {self.type} not implemented")