import logging
from typing import Any

import networkx as nx
from torch_geometric.data import Data

from ApexDAG.nn.data.v2.tensor_encoder import EncoderV2
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class CytoscapeParser:
    """
    Translates raw frontend Cytoscape JSON payloads into
    PyTorch Geometric Data objects using the V2 pipeline.
    """

    def __init__(self, encoder: EncoderV2) -> None:
        self._encoder = encoder

    def _json_to_networkx(self, cyto_elements: list[dict[str, Any]]) -> nx.MultiDiGraph:
        """
        Strips visual frontend metadata and reconstructs the topological DAG.
        """
        graph = nx.MultiDiGraph()

        for element in cyto_elements:
            data = element.get("data", {})
            group = element.get("group", "")

            if group == "nodes":
                node_id = data.get("id")
                if not node_id:
                    continue

                # Extract only the semantic features required by the V2 Encoder
                graph.add_node(
                    node_id,
                    label=str(data.get("label", "")),
                    code=str(data.get("code", "")),
                    node_type=int(data.get("node_type", -1)),
                )

            elif group == "edges":
                source = data.get("source")
                target = data.get("target")
                if not source or not target:
                    continue

                edge_label_str = str(data.get("label", ""))
                ground_truth_label = int(
                    data.get("predicted_label", data.get("edge_type", -1))
                )

                graph.add_edge(
                    source,
                    target,
                    label=edge_label_str,
                    predicted_label=ground_truth_label,
                )

        logger.debug(
            f"Parsed Cytoscape JSON into NetworkX graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges."
        )
        return graph

    def process_annotated_json(self, cyto_elements: list[dict[str, Any]]) -> Data:
        """
        Converts the frontend payload directly into a PyG Data object
        ready for the online training loop.
        """
        # 1. Reconstruct the topological graph
        nx_graph = self._json_to_networkx(cyto_elements)

        # 2. Delegate to V2 Encoder (handles Pruning, CodeBERT, and Tensorization)
        pyg_data = self._encoder.encode(nx_graph)

        return pyg_data
