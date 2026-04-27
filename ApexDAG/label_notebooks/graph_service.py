import networkx as nx

from ApexDAG.label_notebooks.schema import MultiGraphContext, MultiSubgraphContext


class SubgraphExtractor:
    """Service layer responsible for graph topological queries."""

    @staticmethod
    def extract(
        graph: nx.MultiDiGraph,
        source_id: str,
        target_id: str,
        edge_key: str,
        max_depth: int = 1,
    ) -> MultiSubgraphContext:
        """
        Extracts a localized subgraph context around a specific edge to provide
        the LLM with topological awareness without overwhelming the context window.
        """
        if source_id not in graph or target_id not in graph:
            raise ValueError("Source or target node not found in the provided graph.")

        visited_nodes = set()

        def _dfs(current_node: str, current_depth: int) -> None:
            if current_depth > max_depth or current_node in visited_nodes:
                return

            visited_nodes.add(current_node)

            for child in graph.successors(current_node):
                _dfs(child, current_depth + 1)
            for parent in graph.predecessors(current_node):
                _dfs(parent, current_depth + 1)

        _dfs(source_id, 0)
        _dfs(target_id, 0)

        subgraph_view = graph.subgraph(visited_nodes)

        # Leverage the factory method to build the base context,
        # then cast to SubgraphContext
        base_context = MultiGraphContext.from_graph(subgraph_view)

        return MultiSubgraphContext(
            nodes=base_context.nodes,
            edges=base_context.edges,
            edge_of_interest=(source_id, target_id, edge_key),
        )
