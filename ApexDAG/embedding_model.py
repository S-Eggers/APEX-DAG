import fasttext
import torch
from transformers import AutoTokenizer, AutoModel


class EmbeddingModel:
    embedding_dimension = None
    
    def get_embedding(self, sentence: str) -> torch.Tensor:
        raise NotImplementedError()
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> 'EmbeddingModel':
        model_type = model_type.lower()
        if model_type == "fasttext":
            model_path = kwargs.get('model_path', "cc.en.300.bin")
            return FasttextEmbeddingModel(model_path)
        elif model_type == "codebert":
            return CodeBERTEmbeddingModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    
class FasttextEmbeddingModel(EmbeddingModel):
    def __init__(self, model_path: str = "cc.en.300.bin"):
        self._model = fasttext.load_model(model_path)
        self.embedding_dimension = 300
    
    def get_embedding(self, sentence: str) -> torch.Tensor:
        if "\n" in sentence:
            self.logger(f"ERROR: Sentence contains newline character: {sentence}")
            sentence = sentence.replace("\n", " ")
        return torch.tensor(self._model.get_sentence_vector(sentence), dtype=torch.float32)
    
class CodeBERTEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # set the model to evaluation mode
        
        self.embedding_dimension = 768
        
    def get_embedding(self, sentence: str) -> torch.Tensor:
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        last_hidden_state = outputs.last_hidden_state  # [batch_size, sequence_length, hidden_size]
        cls_embedding = last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        return cls_embedding.squeeze()  # remove batch dimension if it's 1
