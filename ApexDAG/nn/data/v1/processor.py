import logging
import traceback
import tqdm
from pathlib import Path
import networkx as nx

from ApexDAG.sca.graph_utils import load_graph
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)

class GraphProcessor:
    """Handles loading and preprocessing of graphs from disk."""
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.graphs = []

    def check_graph(self, G: nx.DiGraph):
        for node, data in G.nodes(data=True):
            for key in data:
                data[key] = "None" if data[key] is None else str(data[key])

        for u, v, data in G.edges(data=True):
            for key in data:
                data[key] = "None" if data[key] is None else str(data[key])

    def load_preprocessed_graphs(self):
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path {self.checkpoint_path} does not exist")

        logger.info("Loading preprocessed graphs...")
        errors = 0
        graph_files = list(self.checkpoint_path.iterdir())

        for graph_file in tqdm.tqdm(graph_files, desc="Loading graphs"):
            try:
                graph = load_graph(graph_file)
                self.graphs.append(graph)
            except Exception:
                logger.error(f"Error in graph {graph_file}\n{traceback.format_exc()}")
                errors += 1

        logger.info(f"Loaded {len(self.graphs)} graphs with {errors} errors")