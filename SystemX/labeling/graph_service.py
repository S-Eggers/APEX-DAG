import networkx as nx

from SystemX.labeling.models import MultiGraphContext, MultiSubgraphContext

class SubgraphExtractor:
    @staticmethod
    def extract(graph: nx.DiGraph, node_id: str, max_depth: int = 1) -> MultiSubgraphContext:
        """Extracts the 'Hub and Spoke' context around an operation node."""
        if node_id not in graph:
            raise ValueError(f"Node {node_id} not found.")

        nodes = nx.single_source_shortest_path_length(graph.to_undirected(), node_id, cutoff=max_depth).keys()

        subgraph_view = graph.subgraph(nodes)
        base_context = MultiGraphContext.from_graph(subgraph_view)

        return MultiSubgraphContext(
            nodes=base_context.nodes,
            edges=base_context.edges,
            node_of_interest=node_id,
        )
