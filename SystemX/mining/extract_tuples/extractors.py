import networkx as nx

from .domain import LineageTuple
from .interfaces import TupleExtractionPolicy

class DAGLineageExtractor(TupleExtractionPolicy):
    """Extracts lineage tuples by resolving state transitions and applying strict topological pruning to eliminate intermediate datasets and combinatorial noise."""

    def extract(self, graph: nx.DiGraph) -> list[LineageTuple]:
        extracted_tuples: set[LineageTuple] = set()

        valid_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("domain_label") != "NOT_RELEVANT"]
        operational_graph = graph.edge_subgraph(valid_edges)

        candidate_roots = set()
        for u, v, d in operational_graph.edges(data=True):
            if d.get("domain_label") == "DATA_IMPORT_EXTRACTION" and graph.nodes[u].get("node_type") not in (6, 7, 8):
                true_root = self._resolve_entity_root(graph, v)
                candidate_roots.add(true_root)

        intermediates = set()
        for root in candidate_roots:
            try:
                descendants = nx.descendants(operational_graph, root)
                intermediates.update(descendants)
            except nx.NetworkXError:
                pass

        true_dataset_roots = candidate_roots - intermediates

        for root_node in true_dataset_roots:
            extracted_tuples.update(self._trace_dataset_lineage(operational_graph, graph, root_node))

        return list(extracted_tuples)

    def _resolve_entity_root(self, graph: nx.DiGraph, node: str) -> str:
        """Traverses backward to find the original instantiation of a variable."""
        current_node = node
        visited = set()

        while current_node not in visited:
            visited.add(current_node)
            current_type = graph.nodes[current_node].get("node_type")

            next_node = None
            for u, _, d in graph.in_edges(current_node, data=True):
                edge_type = d.get("edge_type")
                u_type = graph.nodes[u].get("node_type")

                if edge_type == 2:
                    next_node = u
                    break

                if edge_type == 0 and u_type == current_type and current_type in (2, 4):
                    next_node = u
                    break

            if not next_node or next_node in visited:
                break

            current_node = next_node

        return current_node

    def _trace_dataset_lineage(self, operational_graph: nx.DiGraph, full_graph: nx.DiGraph, dataset_root: str) -> list[LineageTuple]:
        local_tuples: set[LineageTuple] = set()
        has_terminal_state = False

        try:
            reachable_data_nodes = nx.descendants(operational_graph, dataset_root)
        except nx.NetworkXError:
            reachable_data_nodes = set()
        reachable_data_nodes.add(dataset_root)

        for data_node in reachable_data_nodes:
            for _, target_op, out_edge_data in operational_graph.out_edges(data_node, data=True):
                domain = out_edge_data.get("domain_label")

                if domain in ("DATA_EXPORT", "ARTIFACT_EXPORT"):
                    sink_root = self._resolve_entity_root(full_graph, target_op)
                    local_tuples.add(
                        LineageTuple(
                            tuple_type="<D, D>",
                            subject_id=dataset_root,
                            object_id=sink_root,
                        )
                    )
                    has_terminal_state = True

                for caller_node, _, in_edge_data in operational_graph.in_edges(target_op, data=True):
                    if caller_node == data_node:
                        continue

                    caller_node_type = full_graph.nodes[caller_node].get("node_type")

                    if caller_node_type == 4 and in_edge_data.get("edge_type") == 0 and in_edge_data.get("domain_label") in ("MODEL_OPERATION", "MODEL_PREDICTION"):
                        model_root = self._resolve_entity_root(full_graph, caller_node)
                        local_tuples.add(
                            LineageTuple(
                                tuple_type="<M, D>",
                                subject_id=model_root,
                                object_id=dataset_root,
                            )
                        )
                        has_terminal_state = True

        if not has_terminal_state:
            local_tuples.add(LineageTuple(tuple_type="<D, Empty>", subject_id=dataset_root, object_id="Empty"))

        return list(local_tuples)
