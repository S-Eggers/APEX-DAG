import logging
import os
import wandb
from dotenv import load_dotenv

load_dotenv()

def setup_wandb(project_name: str, name: str = None):
    """
    Initialize wandb.

    Args:
        project_name (str): The name of the wandb project.
    """
    entity = os.getenv("WANDB_USER", "default_user")
    
    if name is None:
        wandb.init(project=project_name, entity=entity)
    else:
        wandb.init(project=project_name, entity=entity, name = 'hash_' + str(name).replace("-", "_"))

def setup_logging(name: str, verbose: bool) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.ERROR)
    
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)

    return logger