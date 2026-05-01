from collections.abc import Mapping, Sequence
from enum import Enum, auto
from typing import (
    Final,
    TypedDict,
    TypeVar,
    cast,
)

import networkx as nx
from ApexDAG.sca import DOMAIN_NODE_TYPES

NodeID = TypeVar("NodeID", bound=int | str)


class TransformStep(TypedDict):
    target_node: str
    operation: str
    transform_code: str


class CytoscapeData(TypedDict, total=False):
    """Specific schema for Cytoscape element data payloads."""

    id: str
    source: str
    target: str
    label: str
    node_type: int
    edge_type: int
    predicted_label: int
    transform_history: list[TransformStep]
    base_inputs: str
    cell_id: str


class CytoscapeElement(TypedDict):
    data: CytoscapeData


class VamsaResponse(TypedDict):
    elements: list[CytoscapeElement]


class VamsaMode(Enum):
    WIR = auto()
    LINEAGE = auto()


class EdgePathData(TypedDict):
    operations: list[str]
    edge_type: int
    predicted_label: int
    cell_id: str


class VamsaSerializer:
    """
    Serializer for VAMSA supporting direct Cytoscape export (WIR)
    and reduced lineage visualization (LINEAGE).
    """

    PRIMARY_ANCHORS: Final[set[int]] = {
        DOMAIN_NODE_TYPES["DATASET"],
        DOMAIN_NODE_TYPES["MODEL"],
    }
    AUX_ANCHORS: Final[set[int]] = {
        DOMAIN_NODE_TYPES["LIBRARY"],
        DOMAIN_NODE_TYPES["LITERAL"],
    }
    ALL_ANCHORS: Final[set[int]] = PRIMARY_ANCHORS | AUX_ANCHORS

    def __init__(self, mode: VamsaMode = VamsaMode.WIR) -> None:
        self.mode = mode

    def to_dict(self, graph: nx.DiGraph) -> VamsaResponse:
        """
        Serializes the graph based on the initialized mode.
        """
        if self.mode == VamsaMode.WIR:
            return self._serialize_wir(graph)
        return self._serialize_lineage(graph)

    def _serialize_wir(self, graph: nx.DiGraph) -> VamsaResponse:
        """
        Performs a standard NetworkX to Cytoscape conversion.
        """
        cyto_data = cast(dict[str, list[CytoscapeElement]], nx.cytoscape_data(graph).get("elements", {}))

        nodes = cyto_data.get("nodes", [])
        edges = cyto_data.get("edges", [])
        return {"elements": [*nodes, *edges]}

    def _serialize_lineage(self, graph: nx.DiGraph) -> VamsaResponse:
        """
        Reduces the graph to primary anchors and annotates intermediate transformations.
        """
        working_graph: nx.DiGraph = graph.copy()

        reduced_v = self._extract_anchor_subgraph(working_graph)
        self._integrate_auxiliary_inputs(reduced_v)
        self._collapse_dataset_chains(reduced_v)

        return self._format_to_cytoscape(reduced_v)

    def _get_edge_attributes(self, graph: nx.DiGraph, u: NodeID, v: NodeID) -> Mapping[str, object]:
        """
        Safely extracts edge attributes regardless of Graph type.
        """
        data = graph.get_edge_data(u, v)
        if data is None:
            return {}

        if isinstance(graph, nx.MultiGraph):
            return cast(Mapping[str, object], next(iter(cast(Mapping[int, Mapping[str, object]], data).values())))

        return cast(Mapping[str, object], data)

    def _extract_anchor_subgraph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Builds a skeletal graph connecting anchor nodes via path traversal.
        """
        anchor_graph = nx.DiGraph()
        anchors = [n for n, d in graph.nodes(data=True) if cast(Mapping[str, object], d).get("node_type") in self.ALL_ANCHORS]

        for n in anchors:
            node_data = dict(graph.nodes[n])
            node_data.update({"transform_history": [], "base_inputs": []})
            anchor_graph.add_node(n, **node_data)

        for start_node in anchors:
            for target, path_data in self._find_next_anchors(graph, cast(NodeID, start_node)):
                anchor_graph.add_edge(start_node, target, **path_data)
        return anchor_graph

    def _find_next_anchors(self, graph: nx.DiGraph, start: NodeID) -> list[tuple[NodeID, EdgePathData]]:
        """
        BFS to locate reachable anchors and capture the intervening path.
        """
        results: list[tuple[NodeID, EdgePathData]] = []
        queue: list[tuple[NodeID, list[str]]] = [(start, [])]
        visited: set[NodeID] = {start}

        while queue:
            curr, path = queue.pop(0)
            for nxt_raw in graph.successors(curr):
                nxt = cast(NodeID, nxt_raw)
                if nxt in visited:
                    continue

                edge_attr = self._get_edge_attributes(graph, curr, nxt)
                step_label = cast(str, edge_attr.get("code") or graph.nodes[nxt].get("label", str(nxt)))
                new_path = [*path, step_label] if step_label else path

                node_type = cast(int, graph.nodes[nxt].get("node_type"))
                if node_type in self.ALL_ANCHORS:
                    results.append(
                        (
                            nxt,
                            {
                                "operations": new_path,
                                "edge_type": cast(int, edge_attr.get("edge_type", 2)),
                                "predicted_label": cast(int, edge_attr.get("predicted_label", 2)),
                                "cell_id": cast(str, edge_attr.get("cell_id", "unknown")),
                            },
                        )
                    )
                else:
                    visited.add(nxt)
                    queue.append((nxt, new_path))
        return results

    def _integrate_auxiliary_inputs(self, graph: nx.DiGraph) -> None:
        """
        Merges LIBRARY and LITERAL nodes into the base_inputs of their successors.
        """
        aux_nodes = [n for n in graph.nodes if cast(int, graph.nodes[n].get("node_type")) in self.AUX_ANCHORS]

        for aux in aux_nodes:
            label = cast(str, graph.nodes[aux].get("label", str(aux)))
            is_lib = cast(int, graph.nodes[aux].get("node_type")) == DOMAIN_NODE_TYPES["LIBRARY"]

            for succ in list(graph.successors(aux)):
                edge_data = graph.edges[aux, succ]
                ops = cast(Sequence[str], edge_data.get("operations", []))
                op_chain = " -> ".join(filter(None, ops))

                formatted_val = f"{label}.{op_chain}" if is_lib and op_chain else (op_chain or label)
                if not is_lib:
                    formatted_val = f'"{formatted_val}"'

                cast(list[str], graph.nodes[succ]["base_inputs"]).append(formatted_val)
            graph.remove_node(aux)

    def _collapse_dataset_chains(self, graph: nx.DiGraph) -> None:
        """
        Collapses sequential DATASET nodes into a single node with history.
        """
        changed = True
        while changed:
            changed = False
            for u in list(graph.nodes):
                if u not in graph or cast(int, graph.nodes[u].get("node_type")) != DOMAIN_NODE_TYPES["DATASET"]:
                    continue

                successors = list(graph.successors(u))
                if len(successors) == 1:
                    v = successors[0]
                    is_dataset = cast(int, graph.nodes[v].get("node_type")) == DOMAIN_NODE_TYPES["DATASET"]
                    is_linear = len(list(graph.predecessors(v))) == 1

                    if is_dataset and is_linear:
                        self._merge_dataset_nodes(graph, u, v)
                        changed = True
                        break

    def _merge_dataset_nodes(self, graph: nx.DiGraph, u: object, v: object) -> None:
        """
        Helper to merge a dataset node (v) into its predecessor (u).
        """
        edge_data = graph.edges[u, v]
        ops_list = cast(Sequence[str], edge_data.get("operations", []))
        ops_str = " -> ".join(filter(None, ops_list)) or "Transform"

        v_inputs = cast(list[str], graph.nodes[v].get("base_inputs", []))
        input_suffix = f" [Inputs: {', '.join(v_inputs)}]" if v_inputs else ""

        target_label = cast(str, graph.nodes[v].get("label", str(v)))

        cast(list[TransformStep], graph.nodes[u]["transform_history"]).append({"target_node": target_label, "operation": f"{ops_str}{input_suffix}", "transform_code": ""})

        v_history = cast(list[TransformStep], graph.nodes[v].get("transform_history", []))
        cast(list[TransformStep], graph.nodes[u]["transform_history"]).extend(v_history)

        for succ in list(graph.successors(v)):
            graph.add_edge(u, succ, **graph.edges[v, succ])
        graph.remove_node(v)

    def _format_to_cytoscape(self, graph: nx.DiGraph) -> VamsaResponse:
        """
        Converts the processed graph into a Cytoscape-compatible dictionary.
        """
        elements: list[CytoscapeElement] = []

        for n, d in graph.nodes(data=True):
            data_map = cast(Mapping[str, object], d)

            payload: CytoscapeData = {
                "id": str(n),
                "label": cast(str, data_map.get("label", str(n))),
                "node_type": 0,
                "transform_history": cast(list[TransformStep], data_map.get("transform_history", [])),
                "base_inputs": ", ".join(cast(list[str], data_map.get("base_inputs", []))),
            }

            # Add remaining metadata excluding strictly handled keys
            for k, v in data_map.items():
                if k not in ("transform_history", "base_inputs", "label", "id"):
                    payload[k] = v  # type: ignore

            elements.append({"data": payload})

        for u, v, d in graph.edges(data=True):
            edge_map = cast(Mapping[str, object], d)
            ops = cast(Sequence[str], edge_map.get("operations", []))
            label = " -> ".join(filter(None, ops)) or "Transform"

            payload: CytoscapeData = {
                "id": f"edge_{u}_{v}",
                "source": str(u),
                "target": str(v),
                "label": label,
                "edge_type": cast(int, edge_map.get("edge_type", 2)),
                "predicted_label": cast(int, edge_map.get("predicted_label", 2)),
            }

            for k, val in edge_map.items():
                if k not in ("operations", "label", "source", "target", "id"):
                    payload[k] = val

            elements.append({"data": payload})

        return {"elements": elements}
