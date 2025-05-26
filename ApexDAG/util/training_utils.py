import random
import numpy as np
import torch
from enum import Enum
import os

DOMAIN_LABEL_TO_SUBSAMPLE = "DATA_TRANSFORM"

TASKS_PER_GRAPH_TRANSFORM_MODE_FINETUNE = {
    "reversed": ["node_classification"],
    "original": ["edge_classification"],
    "reversed_masked": ["node_classification", "reconstruction"],
    "original_masked": ["edge_classification", "reconstruction"]
}
TASKS_PER_GRAPH_TRANSFORM_MODE_PRETRAIN = {
    "reversed": ["node_classification", "edge_classification"],
    "original": ["node_classification", "edge_classification", "edge_existence"],
    "reversed_masked": ["node_classification", "edge_classification", "reconstruction"],
    "original_masked": ["node_classification", "edge_classification", "reconstruction", "edge_existence"]
}
class GraphTransformsMode(Enum):
    """Modes for Experiments"""
    REVERSED = "reversed"
    ORIGINAL = "original"
    REVERSED_MASKED = "reversed_masked"
    ORIGINAL_MASKED = "original_masked"
    
class InsufficientNegativeEdgesException(Exception):
    def __init__(self, message="The graph does not have enough negative edges"):
        self.message = message
        super().__init__(self.message)

class InsufficientPositiveEdgesException(Exception):
    def __init__(self, message="The graph does not have enough negative edges"):
        self.message = message
        super().__init__(self.message)      

def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False