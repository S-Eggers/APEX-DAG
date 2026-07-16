from typing import Protocol

import networkx as nx

from SystemX.sca import NODE_TYPES

class ProvenancePolicy(Protocol):
    def apply(self, graph: nx.MultiDiGraph) -> None:
        """Applies provenance enrichment to the graph in-place."""
        ...

class NoProvenancePolicy:
    def apply(self, graph: nx.MultiDiGraph) -> None:
        pass

class FixedPointProvenanceEnricher:
    """Mutates the underlying MultiDiGraph in-place to calculate transitive dataflow closure."""

    def __init__(self, max_depth: int = 15) -> None:
        self._max_depth = max_depth
        self._compute_hubs: set[int] = {NODE_TYPES.get("FUNCTION", 3), NODE_TYPES.get("LOOP", 6), NODE_TYPES.get("CALL", 9)}

    def apply(self, graph: nx.MultiDiGraph) -> None:
        self._initialize_state(graph)
        self._propagate_provenance(graph)
        self._cleanup_and_finalize(graph)

    def _initialize_state(self, graph: nx.MultiDiGraph) -> None:
        for node in graph.nodes():
            graph.nodes[node]["transform_history"] = []
            graph.nodes[node]["_temp_base_inputs"] = []

    def _propagate_provenance(self, graph: nx.MultiDiGraph) -> None:
        changed = True
        iteration = 0

        while changed and iteration < self._max_depth:
            changed = False
            iteration += 1

            for node in graph.nodes():
                data = graph.nodes[node]
                node_type = data.get("node_type")

                hist_len = len(data["transform_history"])
                base_len = len(data["_temp_base_inputs"])

                for pred in graph.predecessors(node):
                    pred_data = graph.nodes[pred]
                    self._merge_node_state(data, pred_data, node_type, str(node), str(pred))

                if len(data["transform_history"]) > hist_len or len(data["_temp_base_inputs"]) > base_len:
                    changed = True

    def _merge_node_state(self, data: dict, pred_data: dict, node_type: int | None, target_id: str, pred_id: str) -> None:
        for step in pred_data.get("transform_history", []):
            if step not in data["transform_history"]:
                data["transform_history"].append(step)

        for b_inp in pred_data.get("_temp_base_inputs", []):
            if b_inp not in data["_temp_base_inputs"]:
                data["_temp_base_inputs"].append(b_inp)

        if pred_data.get("node_type") == NODE_TYPES.get("IMPORT", 1):
            lib_name = pred_data.get("label", pred_id)
            if lib_name not in data["_temp_base_inputs"]:
                data["_temp_base_inputs"].append(lib_name)

        if pred_data.get("node_type") in self._compute_hubs and node_type not in self._compute_hubs:
            step = {"operation": str(pred_data.get("label", "operation")), "target_node": str(data.get("label", target_id)), "transform_code": str(pred_data.get("code", ""))}
            if step not in data["transform_history"]:
                data["transform_history"].append(step)

    def _cleanup_and_finalize(self, graph: nx.MultiDiGraph) -> None:
        for node in graph.nodes():
            data = graph.nodes[node]
            if data.get("_temp_base_inputs"):
                data["base_inputs"] = ", ".join(sorted(data["_temp_base_inputs"]))
            data.pop("_temp_base_inputs", None)
