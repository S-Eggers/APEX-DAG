import os
import tqdm
import torch
import logging
import traceback
import wandb
from pathlib import Path
from torch.utils.data import random_split, Subset
from enum import Enum
from ApexDAG.encoder import Encoder
from ApexDAG.sca.graph_utils import load_graph
from ApexDAG.nn.dataset import GraphDataset
from ApexDAG.nn.training.pretraining_trainer import PretrainingTrainer, PretrainingTrainerMasked
from ApexDAG.nn.training.finetuning_trainer import FinetuningTrainer
from ApexDAG.util.training_utils import InsufficientNegativeEdgesException, InsufficientPositiveEdgesException, GraphTransformsMode, DOMAIN_LABEL_TO_SUBSAMPLE
from ApexDAG.sca.constants import DOMAIN_EDGE_TYPES


class Modes(Enum):
    PRETRAINING = "pretraining"
    FINETUNING = "finetune"
    
class GraphProcessor:
    """Handles loading and preprocessing of graphs."""
    def __init__(self, checkpoint_path: Path, logger: logging.Logger):
        self.checkpoint_path = checkpoint_path
        self.logger = logger
        self.graphs = []

    def check_graph(self, G):
        """Ensures graph attributes are properly formatted."""
        for node, data in G.nodes(data=True):
            for key in data:
                data[key] = "None" if data[key] is None else str(data[key])

        for u, v, data in G.edges(data=True):
            for key in data:
                data[key] = "None" if data[key] is None else str(data[key])

    def load_preprocessed_graphs(self):
        """Loads preprocessed graphs from the checkpoint path."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path {self.checkpoint_path} does not exist")

        self.logger.info("Loading preprocessed graphs...")
        errors = 0
        graph_files = list(self.checkpoint_path.iterdir())
        
        for graph_file in tqdm.tqdm(graph_files, desc="Loading graphs"):
            try:
                graph = load_graph(graph_file)
                self.graphs.append(graph)
            except Exception:
                self.logger.error(f"Error in graph {graph_file}")
                self.logger.error(traceback.format_exc())
                errors += 1

        self.logger.info(f"Loaded {len(self.graphs)} graphs with {errors} errors")

class GraphEncoder:
    """Handles encoding of graphs into tensors for training."""
    def __init__(self, encoded_checkpoint_path: Path, 
                 logger: logging.Logger, 
                 min_nodes: int, 
                 min_edges: int, 
                 load_encoded_old_if_exist: bool,
                 mode: str = "original",
                 bidirectional: bool = False): # add bidirectionality experiment!
        
        self.bidirectional = bidirectional
        self.mode = mode
        self.encoded_checkpoint_path = encoded_checkpoint_path
        self.logger = logger
        self.encoded_graphs = []
        
        # hyperparams
        self.min_nodes = min_nodes
        self.min_edges = min_edges
        
        # for testing, remove if not needed downstream
        self.load_old_if_exist = load_encoded_old_if_exist
    
    def reload_encoded_graphs(self):
        if self.encoded_checkpoint_path.exists() and self.load_old_if_exist:
            self.logger.info("Loading encoded graphs...")
            return [
                torch.load(self.encoded_checkpoint_path / path)
                for path in tqdm.tqdm(os.listdir(self.encoded_checkpoint_path), desc="Loading encoded graphs")
            ]
        return False

    def _make_bidirectional(self, graph):
        """Makes the graph bidirectional by adding reverse edges.
           - should improve message passing scheme.
        """
        for u, v in list(graph.edges()):
            if not graph.has_edge(v, u):
                graph.add_edge(v, u) 
        return graph
    
    def encode_graphs(self, graphs, feature_to_encode):
        """Encodes graphs and saves them to disk."""
        self.logger.info("Encoding graphs...")
        os.makedirs(self.encoded_checkpoint_path, exist_ok=True)
        encoder = Encoder()

        for index, graph in tqdm.tqdm(enumerate(graphs), desc="Encoding graphs"):
            if len(graph.nodes) < self.min_nodes and len(graph.edges) < self.min_edges:
                continue  # skip small graphs
            
            if self.bidirectional:
                graph = self._make_bidirectional(graph)
                          
            try:
                if self.mode in [GraphTransformsMode.REVERSED, GraphTransformsMode.REVERSED_MASKED]:
                    encoded_graph = encoder.encode_reversed(graph, feature_to_encode)
                else:
                    encoded_graph = encoder.encode(graph, feature_to_encode)
                torch.save(encoded_graph, self.encoded_checkpoint_path / f"graph_{index}.pt")
                self.encoded_graphs.append(encoded_graph)
            except KeyboardInterrupt:
                self.logger.warning("Encoding interrupted, continuing with next graph...")
                continue
            except InsufficientNegativeEdgesException:
                self.logger.error(f"Insufficient negative edges in graph {index}")
                continue
            except InsufficientPositiveEdgesException:
                self.logger.error(f"Insufficient positive edges in graph {index}")
                continue
            except Exception:
                self.logger.error(f"Error in graph {index}")


        return self.encoded_graphs


class GATTrainer:
    """Handles model training."""
    def __init__(self, config, logger: logging.Logger, mode: Modes):
        self.config = config
        self.mode = mode
        self.logger = logger
        self.subsample_train = config.get("subsample", False)
        self.graph_transform_mode = config.get("mode", "ORIGINAL")
        self.subsample_thereshold = self.config.get("subsample_thereshold", 0.999999)
        self.trainer = None
        
    def split_and_subsample(self, dataset, split_ratio, subsample = False):
        """Splits the dataset into train and validation sets and subsamples the training set."""
        train_dataset, val_dataset = random_split(dataset, split_ratio)
        
        if subsample and self.mode == Modes.FINETUNING:
            index_to_subsample = DOMAIN_EDGE_TYPES[DOMAIN_LABEL_TO_SUBSAMPLE]
            
            if self.graph_transform_mode in [GraphTransformsMode.REVERSED, GraphTransformsMode.REVERSED_MASKED]:
                filtered_indices = [
                    idx for idx in train_dataset.indices
                    if  sum(dataset[idx].node_types == index_to_subsample)/len(dataset[idx].node_types) < self.subsample_thereshold
                ]
            elif self.graph_transform_mode in [GraphTransformsMode.ORIGINAL, GraphTransformsMode.ORIGINAL_MASKED]:
                filtered_indices = [
                    idx for idx in train_dataset.indices
                    if sum(dataset[idx].edge_types == index_to_subsample)/len(dataset[idx].edge_types) < self.subsample_thereshold
                ]
            train_dataset = Subset(dataset, filtered_indices)

        return train_dataset, val_dataset

    def train(self, encoded_graphs, model, device:str = "cuda", graph_transform_mode: str = GraphTransformsMode.ORIGINAL):
        """Trains the GAT model."""
        wandb.watch(model, log="all")
        
        dataset = GraphDataset(encoded_graphs)
        
        if self.mode == Modes.PRETRAINING:
            self.logger.info("Training in pretraining mode")
            train_size = int(self.config["train_split"] * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
            self.trainer = PretrainingTrainer(model, train_dataset, val_dataset, device=device, patience=self.config["patience"], batch_size=self.config["batch_size"], lr = self.config['learning_rate'], weight_decay = self.config['weight_decay'], graph_transform_mode = graph_transform_mode, logger = self.logger)
            if graph_transform_mode in [GraphTransformsMode.REVERSED_MASKED, GraphTransformsMode.ORIGINAL_MASKED]:
                self.trainer = PretrainingTrainerMasked(model, train_dataset, val_dataset, device=device, patience=self.config["patience"], batch_size=self.config["batch_size"], lr = self.config['learning_rate'], weight_decay = self.config['weight_decay'], graph_transform_mode = graph_transform_mode, logger = self.logger)
        elif self.mode == Modes.FINETUNING:
            self.logger.info("Training in finetuning mode")
            
            train_size = int(self.config["train_split"] * len(dataset))
            test_size = int(self.config["test_split"] * len(dataset))
            val_size = len(dataset) - train_size - test_size
            
            train_dataset, val_dataset = self.split_and_subsample(dataset, [train_size, val_size + test_size], subsample = self.subsample_train)
            val_dataset, test_dataset = random_split(val_dataset, [val_size, test_size])
            
            self.trainer = FinetuningTrainer(model, train_dataset, val_dataset, test_dataset, device=device, patience=self.config["patience"], batch_size=self.config["batch_size"], lr = self.config['learning_rate'], weight_decay = self.config['weight_decay'], graph_transform_mode = graph_transform_mode, logger = self.logger)
        best_loss = self.trainer.train(num_epochs=self.config["num_epochs"])
        
        for type_conf_matrix in self.trainer.conf_matrices_types:
            self.trainer.log_confusion_matrix(train_dataset, "Train", type_conf_matrix )
            self.trainer.log_confusion_matrix(val_dataset, "Val", type_conf_matrix)
            if self.mode == Modes.FINETUNING:
                self.trainer.log_confusion_matrix(test_dataset, "Test", type_conf_matrix)
                
        return best_loss