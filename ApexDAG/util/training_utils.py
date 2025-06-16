import random
import numpy as np
import torch
from enum import Enum
import os

DOMAIN_LABEL_TO_SUBSAMPLE = "DATA_TRANSFORM"

TASKS_PER_GRAPH_TRANSFORM_MODE_FINETUNE = {
    "REVERSED": ["node_classification"],
    "ORIGINAL": ["edge_classification"],
    "REVERSED_MASKED": ["node_classification", "reconstruction"],
    "ORIGINAL_MASKED": ["edge_classification", "reconstruction"]
}
TASKS_PER_GRAPH_TRANSFORM_MODE_PRETRAIN = {
    "REVERSED": ["node_classification", "edge_classification"],
    "ORIGINAL": ["node_classification", "edge_classification", "edge_existence"],
    "REVERSED_MAKSED": ["node_classification", "edge_classification", "reconstruction"],
    "ORIGINAL_MASKED": ["node_classification", "edge_classification", "reconstruction", "edge_existence"]
}
class GraphTransformsMode(Enum):
    """Modes for Experiments"""
    REVERSED = "REVERSED"
    ORIGINAL = "ORIGINAL"
    REVERSED_MASKED = "REVERSED_MASKED"
    ORIGINAL_MASKED = "ORIGINAL_MASKED"
    
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