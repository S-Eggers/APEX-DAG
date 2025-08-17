import unittest
from unittest.mock import MagicMock, patch
import torch
from google.genai import types
from ApexDAG.embedding import Embedding, EmbeddingType

class TestEmbedding(unittest.TestCase):

    def setUp(self):
        self.mock_logger = MagicMock()

    def test_embedding_type_enum(self):
        self.assertIn(EmbeddingType.FASTTEXT, EmbeddingType)
        self.assertIn(EmbeddingType.GEMINI_STANDARD, EmbeddingType)
        self.assertIn(EmbeddingType.GEMINI_CODE, EmbeddingType)
        self.assertIn(EmbeddingType.GRAPHCODEBERT, EmbeddingType)
        self.assertIn(EmbeddingType.CODEBERT, EmbeddingType)
        self.assertIn(EmbeddingType.BERT, EmbeddingType)

    @patch('fasttext.load_model')
    def test_fasttext_init(self, mock_load_model):
        mock_load_model.return_value = MagicMock()
        embedding = Embedding(EmbeddingType.FASTTEXT, self.mock_logger)
        mock_load_model.assert_called_once_with("cc.en.300.bin")
        self.assertEqual(embedding._embedding_model, "cc.en.300.bin")

    @patch('google.genai.Client')
    def test_gemini_init(self, mock_genai_client):
        mock_client_instance = MagicMock()
        mock_genai_client.return_value = mock_client_instance
        # No need to mock embed_content here, as it's mocked in the individual embed tests

        embedding_standard = Embedding(EmbeddingType.GEMINI_STANDARD, self.mock_logger)
        self.assertEqual(embedding_standard._embedding_model, "gemini-embedding-001")

        embedding_code = Embedding(EmbeddingType.GEMINI_CODE, self.mock_logger)
        self.assertEqual(embedding_code._embedding_model, "gemini-embedding-001")

    def test_unimplemented_embedding_type_init(self):
        embedding = Embedding(EmbeddingType.GRAPHCODEBERT, self.mock_logger)
        self.assertIsNone(embedding._model)

    @patch('fasttext.load_model')
    def test_embed_fasttext(self, mock_load_model):
        mock_model = MagicMock()
        mock_model.get_sentence_vector.return_value = [0.1, 0.2, 0.3]
        mock_load_model.return_value = mock_model

        embedding = Embedding(EmbeddingType.FASTTEXT, self.mock_logger)
        result = embedding.embed("test sequence")

        mock_model.get_sentence_vector.assert_called_once_with("test sequence")
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.equal(result, torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)))

    @patch('google.genai.Client')
    def test_embed_gemini_standard(self, mock_genai_client):
        mock_model = MagicMock()
        mock_genai_client.return_value = mock_model
        mock_model.models.embed_content.return_value = MagicMock(embeddings=[0.4, 0.5, 0.6]) # Note: 'embeddings' not 'embedding'

        embedding = Embedding(EmbeddingType.GEMINI_STANDARD, self.mock_logger)
        result = embedding.embed("test sequence")

        mock_model.models.embed_content.assert_called_once_with(model="gemini-embedding-001", contents="test sequence", config=types.EmbedContentConfig(task_type="CLASSIFICATION"))
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.equal(result, torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32)))

    @patch('google.genai.Client')
    def test_embed_gemini_code(self, mock_genai_client):
        mock_model = MagicMock()
        mock_genai_client.return_value = mock_model
        mock_model.models.embed_content.return_value = MagicMock(embeddings=[0.7, 0.8, 0.9]) # Note: 'embeddings' not 'embedding'

        embedding = Embedding(EmbeddingType.GEMINI_CODE, self.mock_logger)
        result = embedding.embed("test sequence")

        mock_model.models.embed_content.assert_called_once_with(model="gemini-embedding-001", contents="test sequence", config=types.EmbedContentConfig(task_type="CODE_RETRIEVAL_QUERY"))
        self.assertIsInstance(result, torch.Tensor)
        self.assertTrue(torch.equal(result, torch.tensor([0.7, 0.8, 0.9], dtype=torch.float32)))

    def test_embed_with_newline(self):
        embedding = Embedding(EmbeddingType.FASTTEXT, self.mock_logger)
        embedding._model = MagicMock()
        embedding._model.get_sentence_vector.return_value = [0.1, 0.2, 0.3]
        
        sequence_with_newline = "test\nsequence"
        embedding.embed(sequence_with_newline)
        embedding._model.get_sentence_vector.assert_called_once_with("test sequence")
        self.mock_logger.warning.assert_called_once()

    def test_embed_unimplemented_type(self):
        embedding = Embedding(EmbeddingType.GRAPHCODEBERT, self.mock_logger)
        with self.assertRaises(NotImplementedError):
            embedding.embed("test sequence")

if __name__ == '__main__':
    unittest.main()
