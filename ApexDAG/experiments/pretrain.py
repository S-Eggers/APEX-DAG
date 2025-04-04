import yaml
import signal
import logging
import wandb
import torch

from pathlib import Path
from ApexDAG.nn.gat import MultiTaskGAT
from ApexDAG.nn.training import GraphProcessor, GraphEncoder, GATTrainer, Modes
from ApexDAG.util.training_utils import set_seed

def create_model(config):
    return MultiTaskGAT(
            hidden_dim=config["hidden_dim"], 
            dim_embed=config["dim_embed"],
            num_heads=config["num_heads"], 
            node_classes=config["node_classes"], 
            edge_classes=config["edge_classes"],
            residual=config["residual"],
            dropout=config["dropout"],
            number_gat_blocks=config["number_gat_blocks"]
        )
    
def signal_handler(signum, frame):
    """Handles interrupt signals (Ctrl+C)."""
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)

def log_config(config_file):
    artifact = wandb.Artifact("config_file", type="config")
    artifact.add_file(config_file)
    wandb.log_artifact(artifact)
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        
    wandb.config.update(config)

def pretrain_gat(args, logger: logging.Logger) -> None:
    """Main entry point for pretraining the GAT model."""
    
    mode = Modes.PRETRAINING
    
    config_path = args.get('config_path')
    log_config(config_path)
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
        
    set_seed(config["seed"])
    checkpoint_path = Path(config["checkpoint_path"])
    encoded_checkpoint_path = Path(config["encoded_checkpoint_path"])

    graph_processor = GraphProcessor(checkpoint_path, logger)
    graph_encoder = GraphEncoder(encoded_checkpoint_path, logger, 
                                 config['min_nodes'], 
                                 config['min_edges'], 
                                 config['load_encoded_old_if_exist'])
    
    model = create_model(config)
    model.to(config['device'])
    
    trainer = GATTrainer(config, logger)

    encoded_graphs = graph_encoder.reload_encoded_graphs()
    
    if not encoded_graphs:
        graph_processor.load_preprocessed_graphs()
        encoded_graphs = graph_encoder.encode_graphs(graph_processor.graphs, feature_to_encode="edge_type")

    best_val_loss = trainer.train(encoded_graphs, model, mode, device=config['device'])
    
    torch.cuda.empty_cache()
    
    return best_val_loss
    
