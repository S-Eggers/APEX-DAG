import logging

import networkx as nx

logger = logging.getLogger(__name__)

class NullPruner:
    """No-op pruner - returns the graph unchanged."""

    def prune(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        return graph

class GraphPruner:
    def __init__(self, protected_node_types: list[int] | None = None) -> None:
        self.protected_node_types = protected_node_types or []

    def prune(self, graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        pruned = graph.copy()
        nodes_removed = True
        iteration = 0

        logger.debug("Starting prune: %d nodes, %d edges.", pruned.number_of_nodes(), pruned.number_of_edges())

        while nodes_removed:
            nodes_removed = False
            iteration += 1

            out_degrees = dict(pruned.out_degree())
            dead_ends = [node for node, out_deg in out_degrees.items() if out_deg == 0]
            nodes_to_delete = []

            for node in dead_ends:
                node_type = int(pruned.nodes[node].get("node_type", -1))
                if node_type not in self.protected_node_types:
                    nodes_to_delete.append(node)

            if nodes_to_delete:
                pruned.remove_nodes_from(nodes_to_delete)
                nodes_removed = True

        isolated_nodes = list(nx.isolates(pruned))
        if isolated_nodes:
            pruned.remove_nodes_from(isolated_nodes)

        logger.info(
            "Pruning complete in %d passes. Result: %d nodes (-%d), %d edges (-%d).",
            iteration,
            pruned.number_of_nodes(),
            graph.number_of_nodes() - pruned.number_of_nodes(),
            pruned.number_of_edges(),
            graph.number_of_edges() - pruned.number_of_edges(),
        )

        return pruned
