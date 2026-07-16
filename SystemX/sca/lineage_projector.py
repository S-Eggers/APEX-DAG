import collections
from typing import TypedDict

import networkx as nx

from SystemX.sca import DOMAIN_EDGE_TYPES, DOMAIN_NODE_TYPES, NODE_TYPES
from SystemX.sca.constants import COMPUTE_HUBS

type NodeId = str

_CONSUMER_TYPES = COMPUTE_HUBS | {NODE_TYPES["INTERMEDIATE"]}

class TransformStep(TypedDict):
    target_node: str
    operation: str
    transform_code: str

class SerializedElementData(TypedDict, total=False):
    id: str
    label: str
    node_type: int
    transform_history: list[TransformStep]
    base_inputs: str
    source: str
    target: str
    edge_type: int
    raw_code: str
    operations: list[str]
    predicted_label: int

class SerializedElement(TypedDict):
    data: SerializedElementData

class SerializedLineage(TypedDict):
    elements: list[SerializedElement]

class BipartiteLineageProjector:
    def __init__(self) -> None:
        self.primary_anchors = {DOMAIN_NODE_TYPES["DATASET"], DOMAIN_NODE_TYPES["MODEL"]}
        self.aux_anchors = {DOMAIN_NODE_TYPES["LIBRARY"], DOMAIN_NODE_TYPES["LITERAL"]}

    def project(self, granular_graph: nx.MultiDiGraph) -> nx.DiGraph:
        macro_graph = nx.DiGraph()

        self._initialize_anchors(granular_graph, macro_graph)
        self._route_bipartite_edges(granular_graph, macro_graph)
        self._collapse_auxiliary_nodes(macro_graph)
        self._collapse_linear_paths(macro_graph)

        return macro_graph

    def _initialize_anchors(self, granular_graph: nx.MultiDiGraph, macro_graph: nx.DiGraph) -> None:
        valid_anchors = self.primary_anchors | self.aux_anchors
        model_type = DOMAIN_NODE_TYPES["MODEL"]
        for n, data in granular_graph.nodes(data=True):
            node_type = data.get("node_type")
            if node_type not in valid_anchors:
                continue
            if node_type == model_type and not data.get("domain_node"):
                continue
            macro_graph.add_node(n, **data)
            macro_graph.nodes[n]["transform_history"] = []
            macro_graph.nodes[n]["base_inputs"] = []
            label = data.get("label", str(n))
            occurrences = [{"cell_id": data.get("cell_id"), "label": label}]
            for succ in granular_graph.successors(n):
                if granular_graph.nodes[succ].get("node_type") in _CONSUMER_TYPES:
                    succ_cell = granular_graph.nodes[succ].get("cell_id")
                    if succ_cell:
                        occurrences.append({"cell_id": succ_cell, "label": label})
            macro_graph.nodes[n]["occurrences"] = occurrences

    def _route_bipartite_edges(self, granular_graph: nx.MultiDiGraph, macro_graph: nx.DiGraph) -> None:
        """Exploits bipartite structure: Source Anchor -> Operation Hub -> Target Anchor."""
        for start_node in macro_graph.nodes:
            queue = collections.deque([(start_node, [])])
            visited = set()

            while queue:
                curr, path = queue.popleft()

                for nxt in granular_graph.successors(curr):
                    if nxt in visited:
                        continue

                    node_data = granular_graph.nodes[nxt]

                    if node_data.get("node_type") in COMPUTE_HUBS:
                        step_info = str(node_data.get("code", node_data.get("label", nxt)))
                        new_path = [*path, step_info] if step_info else path
                        visited.add(nxt)
                        queue.append((nxt, new_path))
                        continue

                    if nxt in macro_graph.nodes:
                        step_info = granular_graph.nodes[nxt].get("label", str(nxt))
                        new_ops = [*path, step_info] if step_info else path

                        hub_preds = [p for p in granular_graph.predecessors(nxt) if granular_graph.nodes[p].get("node_type") in COMPUTE_HUBS]

                        domain_lbl = 2
                        if hub_preds:
                            domain_lbl = granular_graph.nodes[hub_preds[0]].get("predicted_label", 2)

                        inherited_cell_id = node_data.get("cell_id", "unknown_cell")

                        macro_graph.add_edge(
                            start_node,
                            nxt,
                            operations=new_ops,
                            predicted_label=domain_lbl,
                            cell_id=inherited_cell_id,
                        )
                    else:
                        visited.add(nxt)
                        queue.append((nxt, path))

    def _collapse_auxiliary_nodes(self, macro_graph: nx.DiGraph) -> None:
        aux_nodes = [n for n in macro_graph.nodes if macro_graph.nodes[n].get("node_type") in self.aux_anchors]

        for aux in aux_nodes:
            aux_type = macro_graph.nodes[aux].get("node_type")
            aux_label = macro_graph.nodes[aux].get("label", str(aux))

            for v in list(macro_graph.successors(aux)):
                if macro_graph.nodes[v].get("node_type") in self.primary_anchors:
                    ops = macro_graph.edges[aux, v].get("operations", [])
                    edge_str = " -> ".join([str(o) for o in ops if o])

                    fmt_val = f"{aux_label}.{edge_str}" if aux_type == DOMAIN_NODE_TYPES["LIBRARY"] and edge_str else (edge_str or aux_label)
                    weight = 0 if aux_type == DOMAIN_NODE_TYPES["LIBRARY"] else 1
                    fmt_val = fmt_val if weight == 0 else f'"{fmt_val}"'

                    macro_graph.nodes[v]["base_inputs"].append((weight, fmt_val))

            macro_graph.remove_node(aux)

    def _collapse_linear_paths(self, macro_graph: nx.DiGraph) -> None:
        changed = True
        while changed:
            changed = False
            for u in list(macro_graph.nodes):
                if u not in macro_graph or macro_graph.nodes[u].get("node_type") != DOMAIN_NODE_TYPES["DATASET"]:
                    continue

                dataset_successors = [s for s in macro_graph.successors(u) if macro_graph.nodes[s].get("node_type") == DOMAIN_NODE_TYPES["DATASET"]]

                if len(dataset_successors) == 1:
                    v = dataset_successors[0]
                    dataset_predecessors = [p for p in macro_graph.predecessors(v) if macro_graph.nodes[p].get("node_type") == DOMAIN_NODE_TYPES["DATASET"]]

                    if len(dataset_predecessors) == 1 and u != v:
                        self._merge_linear_nodes(macro_graph, u, v)
                        changed = True
                        break

    def _merge_linear_nodes(self, macro_graph: nx.DiGraph, u: NodeId, v: NodeId) -> None:
        edge_data = macro_graph.edges[u, v]
        ops = edge_data.get("operations", [])
        v_label = macro_graph.nodes[v].get("label", str(v))
        step_desc = " -> ".join([str(o) for o in ops if o]) or "Transform"

        raw_aux = macro_graph.nodes[v].get("base_inputs", [])
        sorted_aux = [item[1] for item in sorted(raw_aux, key=lambda x: x[0])]
        aux_str = f" [Inputs: {', '.join(sorted_aux)}]" if sorted_aux else ""

        edge_label = edge_data.get("predicted_label", DOMAIN_EDGE_TYPES["DATA_TRANSFORM"])
        if edge_label == DOMAIN_EDGE_TYPES["DATA_TRANSFORM"]:
            step: TransformStep = {
                "target_node": v_label,
                "operation": step_desc + aux_str,
                "transform_code": "",
            }
            macro_graph.nodes[u]["transform_history"].append(step)

        macro_graph.nodes[u]["transform_history"].extend(macro_graph.nodes[v].get("transform_history", []))

        macro_graph.nodes[u].setdefault("occurrences", []).extend(macro_graph.nodes[v].get("occurrences", []))

        for v_succ in list(macro_graph.successors(v)):
            edge_data = macro_graph.edges[v, v_succ]
            macro_graph.add_edge(u, v_succ, **edge_data)

        macro_graph.remove_node(v)
