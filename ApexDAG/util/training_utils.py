import random
import numpy as np
import torch

class InsufficientNegativeEdgesException(Exception):
    def __init__(self, message="The graph does not have enough negative edges"):
        self.message = message
        super().__init__(self.message)

def set_seed(seed: int) -> None:
    """Sets the random seed for reproducibility across multiple libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 