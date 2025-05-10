import yaml
import logging
import torch
from pathlib import Path
from ApexDAG.util.training_utils import GraphTransformsMode, set_seed, TASKS_PER_GRAPH_TRANSFORM_MODE_FINETUNE
from ApexDAG.util.logging import setup_wandb

from ApexDAG.nn.training import GraphProcessor, GraphEncoder, GATTrainer, Modes
from ApexDAG.experiments.pretrain import create_model as create_pretrain_model


def create_model(config):
    model = create_pretrain_model(config, tasks =TASKS_PER_GRAPH_TRANSFORM_MODE_FINETUNE[config["mode"].value])

    if config["pretrained_model_path"]:
        pretrained_state_dict = torch.load(config["pretrained_model_path"])
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and "head" not in k}
        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict)
    
    return model
    
def finetune_gat(args, logger: logging.Logger) -> None:
    """Main entry point for tinetuning the GAT model, linear probing of the last layers/heads."""
    
    mode = Modes.FINETUNING
    
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
        graph_transform_mode = config.get("mode", "ORIGINAL")
        config["mode"] = GraphTransformsMode[graph_transform_mode]
        hash_value = hash(str(config))
        
        set_seed(config["seed"])
        setup_wandb(project_name=f"APEX-DAG-{config['mode']}-finetune", name = hash_value)

    checkpoint_path = Path(config["checkpoint_path"])
    encoded_checkpoint_path = Path(config["encoded_checkpoint_path"]).parent / "pytorch-encoded-finetune"

    graph_processor = GraphProcessor(checkpoint_path, logger)
    graph_encoder = GraphEncoder(encoded_checkpoint_path, 
                                 logger, config['min_nodes'], 
                                 config['min_edges'], 
                                 config['load_encoded_old_if_exist'],
                                 mode = config["mode"],
                                 subsample = config.get("subsample", False)
    )
    
    model = create_model(config)
    
    trainer = GATTrainer(config, logger)

    encoded_graphs = False # graph_encoder.reload_encoded_graphs()
    
    if not encoded_graphs:
        graph_processor.load_preprocessed_graphs()
        encoded_graphs = graph_encoder.encode_graphs(graph_processor.graphs, feature_to_encode="domain_label")

    # train model
    trainer.train(encoded_graphs, model, mode, device= config['device'], graph_transform_mode = config["mode"])
