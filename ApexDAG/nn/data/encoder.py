import os
import torch
import logging
import traceback
import tqdm
from pathlib import Path
import networkx as nx

from ApexDAG.encoder import Encoder
from ApexDAG.util.training_utils import (
    InsufficientNegativeEdgesException,
    InsufficientPositiveEdgesException,
    GraphTransformsMode
)

class GraphEncoder:
    """Handles encoding of NetworkX graphs into PyTorch Geometric tensors."""
    def __init__(
        self,
        encoded_checkpoint_path: Path,
        logger: logging.Logger,
        min_nodes: int,
        min_edges: int,
        load_encoded_old_if_exist: bool,
        mode: str = "original",
        bidirectional: bool = False,
    ):
        self.bidirectional = bidirectional
        self.mode = mode
        self.encoded_checkpoint_path = encoded_checkpoint_path
        self.encoded_graphs = []
        self.logger = logger
        self.min_nodes = min_nodes
        self.min_edges = min_edges
        self.load_old_if_exist = load_encoded_old_if_exist

    def reload_encoded_graphs(self):
        if self.encoded_checkpoint_path.exists() and self.load_old_if_exist:
            self.logger.info("Loading encoded graphs...")
            return [
                torch.load(self.encoded_checkpoint_path / path)
                for path in tqdm.tqdm(os.listdir(self.encoded_checkpoint_path), desc="Loading encoded graphs")
            ]
        return False

    def _make_bidirectional(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Makes the graph bidirectional. 
        """
        return graph.to_undirected().to_directed()

    def encode_graphs(self, graphs, feature_to_encode: str):
        self.logger.info("Encoding graphs...")
        os.makedirs(self.encoded_checkpoint_path, exist_ok=True)
        encoder = Encoder(logger=self.logger)

        for index, graph in tqdm.tqdm(enumerate(graphs), desc="Encoding graphs"):
            if len(graph.nodes) < self.min_nodes and len(graph.edges) < self.min_edges:
                continue 

            if self.bidirectional:
                graph = self._make_bidirectional(graph)

            try:
                if self.mode in [GraphTransformsMode.REVERSED.value, GraphTransformsMode.REVERSED_MASKED.value]:
                    encoded_graph = encoder.encode_reversed(graph, feature_to_encode)
                else:
                    encoded_graph = encoder.encode(graph, feature_to_encode)
                    
                torch.save(encoded_graph, self.encoded_checkpoint_path / f"graph_{index}.pt")
                self.encoded_graphs.append(encoded_graph)
                
            except KeyboardInterrupt:
                self.logger.warning("Encoding interrupted, continuing with next graph...")
                continue
            except (InsufficientNegativeEdgesException, InsufficientPositiveEdgesException) as e:
                self.logger.error(f"{e.__class__.__name__} in graph {index}")
                continue
            except Exception:
                self.logger.error(f"Error in graph {index}\n{traceback.format_exc()}")
                continue

        return self.encoded_graphs