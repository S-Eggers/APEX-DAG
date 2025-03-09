import yaml
import logging
import torch
from pathlib import Path
from ApexDAG.nn.gat import MultiTaskGAT

from ApexDAG.nn.training import GraphProcessor, GraphEncoder, GATTrainer, Modes
from ApexDAG.experiments.pretrain import create_model as create_pretrain_model


def create_model(config):
    model = create_pretrain_model(config)

    if config["pretrained_model_path"]:
        pretrained_state_dict = torch.load(config["pretrained_model_path"])
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and "head" not in k}
        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict)
        
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    
    return model
    
def finetune_gat(args, logger: logging.Logger) -> None:
    """Main entry point for tinetuning the GAT model, linear probing of the last layers/heads."""
    
    mode = Modes.LINEAR_PROBING

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    checkpoint_path = Path(config["checkpoint_path"])
    encoded_checkpoint_path = Path(config["encoded_checkpoint_path"]).parent / "pytorch-encoded-finetune"

    graph_processor = GraphProcessor(checkpoint_path, logger)
    graph_encoder = GraphEncoder(encoded_checkpoint_path, logger, config['min_nodes'], config['min_edges'], config['load_encoded_old_if_exist'])
    
    model = create_model(config)
    
    trainer = GATTrainer(config, logger)

    # load or mine graphs
    graph_processor.load_preprocessed_graphs()

    # encode graphs
    encoded_graphs = graph_encoder.encode_graphs(graph_processor.graphs)

    # train model
    trainer.train(encoded_graphs, model, mode)
