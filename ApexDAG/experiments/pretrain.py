import os
import yaml
import tqdm
import torch
import signal
import logging
import traceback
from pathlib import Path
from torch.utils.data import random_split

from ApexDAG.encoder import Encoder
from ApexDAG.sca.graph_utils import load_graph
from ApexDAG.nn.gat import MultiTaskGAT
from ApexDAG.nn.dataset import GraphDataset
from ApexDAG.nn.trainer import PretrainingTrainer


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
        graph_files = list(self.checkpoint_path.iterdir())[:100]
        
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
    def __init__(self, encoded_checkpoint_path: Path, logger: logging.Logger, min_nodes: int, min_edges: int, load_encoded_old_if_exist: bool):
        self.encoded_checkpoint_path = encoded_checkpoint_path
        self.logger = logger
        self.encoded_graphs = []
        
        # hyperparams
        self.min_nodes = min_nodes
        self.min_edges = min_edges
        
        # for testing, remove if not needed downstream
        self.load_old_if_exist = load_encoded_old_if_exist

    def encode_graphs(self, graphs):
        """Encodes graphs and saves them to disk."""
        if self.encoded_checkpoint_path.exists() and self.load_old_if_exist:
            self.logger.info("Loading encoded graphs...")
            self.encoded_graphs = [
                torch.load(self.encoded_checkpoint_path / path)
                for path in tqdm.tqdm(os.listdir(self.encoded_checkpoint_path), desc="Loading encoded graphs")
            ]
        else:
            self.logger.info("Encoding graphs...")
            os.makedirs(self.encoded_checkpoint_path, exist_ok=True)
            encoder = Encoder()

            for index, graph in tqdm.tqdm(enumerate(graphs), desc="Encoding graphs"):
                if len(graph.nodes) < self.min_nodes and len(graph.edges) < self.min_edges:
                    continue  # skip small graphs

                try:
                    encoded_graph = encoder.encode(graph)
                    torch.save(encoded_graph, self.encoded_checkpoint_path / f"graph_{index}.pt")
                    self.encoded_graphs.append(encoded_graph)
                except KeyboardInterrupt:
                    self.logger.warning("Encoding interrupted, continuing with next graph...")
                    continue

        return self.encoded_graphs


class GATTrainer:
    """Handles model training."""
    def __init__(self, config, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def train(self, encoded_graphs):
        """Trains the GAT model."""
        dataset = GraphDataset(encoded_graphs)
        train_size = int(self.config["train_split"] * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        model = MultiTaskGAT(
            hidden_dim=self.config["hidden_dim"], 
            num_heads=self.config["num_heads"], 
            node_classes=self.config["node_classes"], 
            edge_classes=self.config["edge_classes"]
        )
        # TODO: need to add wandb logging (as in issue #38)
        trainer = PretrainingTrainer(model, train_dataset, val_dataset, device="cpu", patience=self.config["patience"])
        trainer.train(num_epochs=self.config["num_epochs"])


class GATPretrainer:
    """Main class to orchestrate the GAT pretraining pipeline."""
    def __init__(self, args):
        self.args = args
        self.logger = self.setup_logger()

        # configuration loading
        with open(args.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.checkpoint_path = Path(self.config["checkpoint_path"])
        self.encoded_checkpoint_path = Path(self.config["encoded_checkpoint_path"]).parent / "pytorch-encoded"

        # processing components
        self.graph_processor = GraphProcessor(self.checkpoint_path, self.logger)
        self.graph_encoder = GraphEncoder(self.encoded_checkpoint_path, self.logger)
        self.trainer = GATTrainer(self.config, self.logger)

    def setup_logger(self):
        """Sets up the logger."""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)

    def run(self):
        """Runs the full training pipeline."""
        # load or mine graphs
        if not self.graph_processor.load_preprocessed_graphs():
            self.graph_processor.mine_dataflows(self.args)

        # encode graphs
        encoded_graphs = self.graph_encoder.encode_graphs(self.graph_processor.graphs)

        # train model
        self.trainer.train(encoded_graphs)


def signal_handler(signum, frame):
    """Handles interrupt signals (Ctrl+C)."""
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)


def pretrain_gat(args, logger: logging.Logger) -> None:
    """Main entry point for pretraining the GAT model."""
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)


    checkpoint_path = Path(config["checkpoint_path"])
    encoded_checkpoint_path = Path(config["encoded_checkpoint_path"]).parent / "pytorch-encoded"

    graph_processor = GraphProcessor(checkpoint_path, logger)
    graph_encoder = GraphEncoder(encoded_checkpoint_path, logger, config['min_nodes'], config['min_edges'], config['load_encoded_old_if_exist'])
    trainer = GATTrainer(config, logger)

    # load or mine graphs
    graph_processor.load_preprocessed_graphs()

    # encode graphs
    encoded_graphs = graph_encoder.encode_graphs(graph_processor.graphs)

    # train model
    trainer.train(encoded_graphs)
