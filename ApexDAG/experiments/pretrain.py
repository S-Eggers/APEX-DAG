import yaml
import signal
import logging
import wandb
from pathlib import Path
from ApexDAG.nn.gat import MultiTaskGAT
from ApexDAG.nn.training import GraphProcessor, GraphEncoder, GATTrainer, Modes


def create_model(config):
    return MultiTaskGAT(
            hidden_dim=config["hidden_dim"], 
            hidden_dim_edge=config["hidden_dim_edge"],
            num_heads=config["num_heads"], 
            node_classes=config["node_classes"], 
            edge_classes=config["edge_classes"],
            hidden_dim_pretrain_edge_embed=config["embed_model_dim"],
            hidden_dim_pretrain_node_embed=config["embed_model_dim"],

        )
    
def signal_handler(signum, frame):
    """Handles interrupt signals (Ctrl+C)."""
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)


def pretrain_gat(args, logger: logging.Logger) -> None:
    """Main entry point for pretraining the GAT model."""
    
    mode = Modes.PRETRAINING
    
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        wandb.config.update(config)
        wandb.save(args.config_path)


    checkpoint_path = Path(config["checkpoint_path"])
    encoded_checkpoint_path = Path(config["encoded_checkpoint_path"]).parent / "pytorch-encoded"

    graph_processor = GraphProcessor(checkpoint_path, logger)
    graph_encoder = GraphEncoder(encoded_checkpoint_path, logger, 
                                 config['min_nodes'], 
                                 config['min_edges'], 
                                 config['load_encoded_old_if_exist'],
                                 embedding_model_name=config['embedding_model_name'])
    
    model = create_model(config)
    
    trainer = GATTrainer(config, logger)

    # load or mine graphs
    graph_processor.load_preprocessed_graphs()

    # encode graphs
    encoded_graphs = graph_encoder.encode_graphs(graph_processor.graphs, feature_to_encode="edge_type")

    # train model
    trainer.train(encoded_graphs, model, mode)
    
    
