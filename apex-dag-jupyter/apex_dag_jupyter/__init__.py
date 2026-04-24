import os
import yaml
import torch
import logging
from pathlib import Path

try:
    from ._version import __version__
except ImportError:
    import warnings
    warnings.warn("Importing 'apex_dag_jupyter' outside a proper installation.")
    __version__ = "dev"

from ApexDAG.nn.models.gat_v1 import MultiTaskGATv1
from ApexDAG.nn.data.encoder import GraphEncoder
from .handlers import setup_handlers
from .config import APEXDAGConfig

def _jupyter_labextension_paths():
    return [{"src": "labextension", "dest": "apex-dag-jupyter"}]

def _jupyter_server_extension_points():
    return [{"module": "apex_dag_jupyter"}]

def create_model(config, reversed_mode, tasks):
    # Enforces the v1 legacy architecture to support existing checkpoints
    return MultiTaskGATv1(
        hidden_dim=config["hidden_dim"],
        dim_embed=config["dim_embed"],
        num_heads=config["num_heads"],
        edge_classes=config["node_classes"] if reversed_mode else config["edge_classes"],
        node_classes=config["edge_classes"] if reversed_mode else config["node_classes"],
        residual=config["residual"],
        dropout=config["dropout"],
        number_gat_blocks=config["number_gat_blocks"],
        task=tasks,
    )

def load_apex_model(logger: logging.Logger):
    """
    Loads the ML model using robust, package-relative path resolution.
    Ideally, these paths should be injected via Jupyter configuration, 
    but this ensures it survives being run from arbitrary directories.
    """
    logger.info("APEX-DAG Plugin: Initializing ML Model...")
    
    # Dynamically resolve the package root directory
    package_root = Path(__file__).parent.parent.absolute()
    
    config_path = package_root / "models" / "config" / "default_reversed.yaml"
    checkpoint_path = package_root / "models" / "checkpoints" / "model_epoch_finetuned_GraphTransformsMode.REVERSED_440.pt"

    if not config_path.exists() or not checkpoint_path.exists():
        logger.error(f"APEX-DAG Plugin Error: Missing required ML assets. Looked in {package_root}/demo/")
        return None

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    encoded_checkpoint_path = Path(config["encoded_checkpoint_path"])

    graph_encoder = GraphEncoder(
        encoded_checkpoint_path,
        logger,
        min_nodes=3,
        min_edges=2,
        load_encoded_old_if_exist=False,
        mode="REVERSED",
    )

    model = create_model(config=config, reversed_mode=True, tasks=["node_classification"])
    logger.info("APEX-DAG Plugin: Loading Model Weights...")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to("cpu")
    model.eval()

    return {"encoder": graph_encoder, "model": model}


def _load_jupyter_server_extension(server_app):
    """Registers the API handler to receive HTTP requests from the frontend extension."""
    apex_model_content = load_apex_model(server_app.log)
    
    if apex_model_content is None:
        server_app.log.warning("APEX-DAG Plugin initialized without ML capabilities due to missing assets.")
    
    setup_handlers(server_app.web_app, apex_model_content, server_app.config)
    name = "apex_dag_jupyter"
    server_app.log.info(f"Registered {name} server extension")