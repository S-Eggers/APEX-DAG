import os
import tqdm
import torch
import logging


def finetune_gat(args, logger: logging.Logger) -> None:
    if args.checkpoint_path is not None:
        model_path = args.checkpoint_path
    else:
        model_path = os.path.join(os.getcwd(), "checkpoints", "final_model.pt")
    
    pass