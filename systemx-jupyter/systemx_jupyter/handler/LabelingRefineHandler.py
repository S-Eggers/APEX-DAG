import json
from typing import Any

import networkx as nx
import tornado.web
from SystemX.sca.refinement.factory import create_default_refiner
from SystemX.serializer.labeling_serializer import LabelingSerializer
from jupyter_server.base.handlers import APIHandler

_UI_KEYS = {"position", "group", "removed", "selected", "selectable", "locked", "grabbable", "pannable", "classes"}
_EDGE_META_KEYS = {"id", "source", "target"}

class _RefineGraphAdapter:
    """Minimal facade exposing the graph and attribute setters the GraphRefiner uses."""

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self._graph = graph

    def get_graph(self) -> nx.MultiDiGraph:
        return self._graph

    def set_domain_node_label(self, attrs: dict, name: str) -> None:
        nx.set_node_attributes(self._graph, attrs, name=name)

    def set_domain_label(self, attrs: dict, name: str) -> None:
        nx.set_edge_attributes(self._graph, attrs, name=name)

def _coerce_elements(raw: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        return raw.get("elements", raw.get("nodes", []) + raw.get("edges", []))
    if isinstance(raw, list):
        return raw
    raise ValueError("Invalid graph payload. Expected list or {elements|nodes|edges}.")

def build_graph_from_elements(elements: list[dict[str, Any]]) -> nx.MultiDiGraph:
    """Rebuild a MultiDiGraph from serialized Cytoscape elements."""
    graph = nx.MultiDiGraph()

    for element in elements:
        data = element.get("data") if isinstance(element, dict) else None
        if not isinstance(data, dict) or data.get("source") is not None:
            continue
        node_id = data.get("id")
        if node_id is None:
            continue
        graph.add_node(node_id, **{k: v for k, v in data.items() if k not in _UI_KEYS and k != "id"})

    for element in elements:
        data = element.get("data") if isinstance(element, dict) else None
        if not isinstance(data, dict):
            continue
        source = data.get("source")
        target = data.get("target")
        if source is None or target is None:
            continue
        graph.add_edge(source, target, **{k: v for k, v in data.items() if k not in _UI_KEYS and k not in _EDGE_META_KEYS})

    return graph

class LabelingRefineHandler(APIHandler):
    """Re-runs only the graph refiner over the current annotation graph."""

    @tornado.web.authenticated
    def post(self) -> None:
        try:
            input_data: dict[str, Any] = self.get_json_body() or {}
            raw_graph = input_data.get("graph", [])

            elements = _coerce_elements(raw_graph)
            graph = build_graph_from_elements(elements)
            adapter = _RefineGraphAdapter(graph)

            create_default_refiner().refine(adapter)

            result = LabelingSerializer().to_dict(adapter)

            self.finish(json.dumps({"success": True, "message": "Refiner applied.", "predictions": result}))

        except ValueError as e:
            self.set_status(400)
            self.finish(json.dumps({"success": False, "message": str(e)}))
        except Exception as e:
            self.log.error(f"Refine error: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Error"}))
