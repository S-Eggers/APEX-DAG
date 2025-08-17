try:
    from ._version import __version__
except ImportError:
    # Fallback when using the package in dev mode without installing
    # in editable mode with pip. It is highly recommended to install
    # the package from a stable release or in editable mode: https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs
    import warnings

    warnings.warn("Importing 'apex_dag_jupyter' outside a proper installation.")
    __version__ = "dev"

import os
import yaml
import torch
import logging
from pathlib import Path
from ApexDAG.nn.gat import MultiTaskGAT
from ApexDAG.nn.training import GraphEncoder, Modes
from ApexDAG.util.training_utils import TASKS_PER_GRAPH_TRANSFORM_MODE_PRETRAIN
from .handlers import setup_handlers
from .config import APEXDAGConfig


def _jupyter_labextension_paths():
    return [{"src": "labextension", "dest": "apex-dag-jupyter"}]


def _jupyter_server_extension_points():
    return [{"module": "apex_dag_jupyter"}]


def create_model(config, reversed, tasks):
    return MultiTaskGAT(
        hidden_dim=config["hidden_dim"],
        dim_embed=config["dim_embed"],
        num_heads=config["num_heads"],
        edge_classes=config["node_classes"] if reversed else config["edge_classes"],
        node_classes=config["edge_classes"] if reversed else config["node_classes"],
        residual=config["residual"],
        dropout=config["dropout"],
        number_gat_blocks=config["number_gat_blocks"],
        task=tasks,
    )


def load_apex_model(config, logger: logging.Logger):
    logger.info("APEX-DAG Plugin: Initializing ML Model...")
    config = yaml.safe_load(
        open(os.path.join("demo", "config", "default_reversed.yaml"), "r")
    )
    encoded_checkpoint_path = Path(config["encoded_checkpoint_path"])

    graph_encoder = GraphEncoder(
        encoded_checkpoint_path,
        logger,
        min_nodes=3,
        min_edges=2,
        load_encoded_old_if_exist=False,
        mode="REVERSED",
    )

    model = create_model(config=config, reversed=True, tasks=["node_classification"])
    logger.info("APEX-DAG Plugin: Loading Model...")
    checkpoint = torch.load(
        os.path.join(
            "demo",
            "checkpoints",
            "model_epoch_finetuned_GraphTransformsMode.REVERSED_440.pt",
        ),
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to("cpu")
    model.eval()

    return {"encoder": graph_encoder, "model": model}


def _load_jupyter_server_extension(server_app):
    """Registers the API handler to receive HTTP requests from the frontend extension.

    Parameters
    ----------
    server_app: jupyterlab.labapp.LabApp
        JupyterLab application instance
    """
    apex_model_content = load_apex_model(server_app.config, server_app.log)
    setup_handlers(server_app.web_app, apex_model_content, server_app.config)
    name = "apex_dag_jupyter"
    server_app.log.info(f"Registered {name} server extension")
