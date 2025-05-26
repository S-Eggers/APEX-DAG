import yaml
import logging
import torch
import time
from tqdm import tqdm
import wandb
from pathlib import Path
from ApexDAG.util.training_utils import GraphTransformsMode, set_seed, TASKS_PER_GRAPH_TRANSFORM_MODE_FINETUNE
from ApexDAG.util.logging import setup_wandb

from ApexDAG.nn.training import GraphProcessor, GraphEncoder, GATTrainer, Modes
from ApexDAG.experiments.pretrain import create_model as create_pretrain_model


def create_model(config):
    model = create_pretrain_model(config, reversed = ('reversed' in config["mode"].value), tasks =TASKS_PER_GRAPH_TRANSFORM_MODE_FINETUNE[config["mode"].value])

    if config["pretrained_model_path"]:
        pretrained_state_dict = torch.load(config["pretrained_model_path"])
        model_state_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict and "head" not in k}
        model_state_dict.update(filtered_state_dict)
        model.load_state_dict(model_state_dict)
    
    return model
    
def check_inference_duration(trainer, logger, encoded_graphs):
    device = torch.device("cpu")
    trainer.trainer.model.to(device)
    trainer.trainer.model.eval()

    encoded_graphs = [graph.to(device) for graph in encoded_graphs]
    start_inference = time.time()
    with torch.no_grad():
        for i, graph in tqdm(enumerate(encoded_graphs)):
            _ = trainer.trainer.model(graph)
    total_inference_time = time.time() - start_inference

    logger.info(f"[{i+1}/{len(encoded_graphs)}] Inference time: {total_inference_time:.4f} seconds")

    avg_time = total_inference_time / len(encoded_graphs)
    logger.info(f"Average inference time per graph: {avg_time:.4f} seconds")

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
        wandb.config.update(config)

    checkpoint_path = Path(config["checkpoint_path"])
    encoded_checkpoint_path = Path(config["encoded_checkpoint_path"]).parent / "pytorch-encoded-finetune"

    graph_processor = GraphProcessor(checkpoint_path, logger)
    graph_encoder = GraphEncoder(encoded_checkpoint_path, 
                                 logger, config['min_nodes'], 
                                 config['min_edges'], 
                                 config['load_encoded_old_if_exist'],
                                 mode = config["mode"],
    )
    
    model = create_model(config)
    
    trainer = GATTrainer(config, logger, mode = mode)

    encoded_graphs = False # graph_encoder.reload_encoded_graphs()
    
    if not encoded_graphs:
        logger.info("Loading and encoding graphs...")
        
        start_load = time.time()
        graph_processor.load_preprocessed_graphs()
        load_duration = time.time() - start_load
        num_graphs = len(graph_processor.graphs)
        logger.info(f"Loaded {num_graphs} graphs in {load_duration:.2f} seconds "
                    f"({load_duration / num_graphs:.4f} s/graph).")

        start_encode = time.time()
        encoded_graphs = graph_encoder.encode_graphs(
            graph_processor.graphs, feature_to_encode="domain_label"
        )
        encode_duration = time.time() - start_encode
        num_encoded = len(encoded_graphs)
        logger.info(f"Encoded {num_encoded} graphs in {encode_duration:.2f} seconds "
                    f"({encode_duration / num_encoded:.4f} s/graph).")

    # train model
    trainer.train(encoded_graphs, model, device= config['device'], graph_transform_mode = config["mode"])
    check_inference_duration(trainer, logger, encoded_graphs)
    wandb.finish()
    
