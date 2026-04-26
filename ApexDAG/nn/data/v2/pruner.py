import logging

import networkx as nx

from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class GraphPruner:
    def __init__(self, protected_node_types: list[int] | None = None) -> None:
        self.protected_node_types = protected_node_types or []

    def prune(self, G: nx.MultiDiGraph) -> nx.MultiDiGraph:
        pruned_G = G.copy()
        nodes_removed = True
        iteration = 0

        logger.debug(
            f"Starting prune: {pruned_G.number_of_nodes()} nodes, {pruned_G.number_of_edges()} edges."
        )

        while nodes_removed:
            nodes_removed = False
            iteration += 1

            out_degrees = dict(pruned_G.out_degree())
            dead_ends = [node for node, out_deg in out_degrees.items() if out_deg == 0]
            nodes_to_delete = []

            for node in dead_ends:
                node_attrs = pruned_G.nodes[node]
                node_type = int(node_attrs.get("node_type", -1))

                if node_type not in self.protected_node_types:
                    nodes_to_delete.append(node)

            if nodes_to_delete:
                pruned_G.remove_nodes_from(nodes_to_delete)
                nodes_removed = True

        isolated_nodes = list(nx.isolates(pruned_G))
        if isolated_nodes:
            pruned_G.remove_nodes_from(isolated_nodes)

        logger.info(
            f"Pruning complete in {iteration} passes. "
            f"Result: {pruned_G.number_of_nodes()} nodes (-{G.number_of_nodes() - pruned_G.number_of_nodes()}), "
            f"{pruned_G.number_of_edges()} edges (-{G.number_of_edges() - pruned_G.number_of_edges()})."
        )

        return pruned_G
