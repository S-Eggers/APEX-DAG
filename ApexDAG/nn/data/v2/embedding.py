import torch
import logging
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class CodeBERTEmbedding:
    """
    V2 Embedding Engine: Utilizes Microsoft's GraphCodeBERT 
    to extract dense semantic vectors from Python AST snippets.
    """
    def __init__(self, model_name: str = "microsoft/graphcodebert-base"):
        self.model_name = model_name
        self._max_output_dim = 768
        
        logger.info(f"Loading HF Tokenizer and Model: {self.model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            logger.info("GraphCodeBERT loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model {self.model_name}: {e}")
            raise

    @property
    def dimension(self) -> int:
        return self._max_output_dim

    @torch.no_grad()
    def embed(self, code_snippet: str) -> torch.Tensor:
        """
        Tokenizes the Python code and passes it through the transformer.
        Returns the [CLS] token embedding as the global semantic representation.
        """
        if not code_snippet or str(code_snippet).strip() == "None":
            return torch.zeros(self.dimension, dtype=torch.float32)

        clean_code = str(code_snippet).replace("\n", " ").strip()

        inputs = self.tokenizer(
            clean_code, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)

        outputs = self.model(**inputs)
        cls_embedding = outputs.last_hidden_state[0, 0, :]
        
        return cls_embedding.cpu()