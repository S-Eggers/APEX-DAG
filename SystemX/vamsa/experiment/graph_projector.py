import logging
from typing import Final

import networkx as nx

from .models import TextSpan
from .spatial_node_matcher import SpatialNodeMatcher

logger = logging.getLogger(__name__)

NodeId = str
EdgeStruct = tuple[NodeId, NodeId]

class GraphProjector:
    """Projects the Vamsa graph onto the Golden dataset topology."""

    def __init__(self, vamsa_graph: nx.DiGraph, matcher: SpatialNodeMatcher) -> None:
        self.vamsa_graph: Final[nx.DiGraph] = vamsa_graph
        self.matcher: Final[SpatialNodeMatcher] = matcher

        self.node_mapping: dict[NodeId, NodeId] = self._build_node_mapping()

    def _build_node_mapping(self) -> dict[NodeId, NodeId]:
        """Iterates over Vamsa nodes and builds the translation dictionary."""
        mapping: dict[NodeId, NodeId] = {}

        for vamsa_node_id, data in self.vamsa_graph.nodes(data=True):
            span = TextSpan(start_line=data.get("lineno"), start_col=data.get("col_offset"), end_line=data.get("end_lineno"), end_col=data.get("end_col_offset"))

            golden_match = self.matcher.find_best_match(str(vamsa_node_id), span)

            if golden_match:
                mapping[str(vamsa_node_id)] = golden_match
            else:
                logger.debug(f"Could not map Vamsa node: {vamsa_node_id} at {span}")

        aligned_count = len(mapping)
        total_count = self.vamsa_graph.number_of_nodes()
        logger.info(f"Node Projection Complete: {aligned_count}/{total_count} mapped.")

        return mapping

    def project_edges(self) -> set[EdgeStruct]:
        """Projects Vamsa edges onto the Golden domain."""
        projected_edges: set[EdgeStruct] = set()

        for u, v in self.vamsa_graph.edges():
            golden_u = self.node_mapping.get(str(u))
            golden_v = self.node_mapping.get(str(v))

            if golden_u and golden_v:
                projected_edges.add((golden_u, golden_v))

        return projected_edges
