from typing import Any

import networkx as nx

from .interfaces import GraphParserPolicy

class CytoscapeGraphParser(GraphParserPolicy):
    def parse(self, raw_data: dict[str, Any]) -> nx.DiGraph:
        graph = nx.DiGraph()
        elements = raw_data.get("elements", [])

        for element in elements:
            data = element.get("data", {})
            if "source" not in data and "target" not in data and "id" in data:
                node_id = data["id"]
                node_attrs = {k: v for k, v in data.items() if k != "id"}
                graph.add_node(node_id, **node_attrs)

        for element in elements:
            data = element.get("data", {})
            if "source" in data and "target" in data:
                source = data["source"]
                target = data["target"]

                edge_attrs = {k: v for k, v in data.items() if k not in ("source", "target")}

                edge_attrs.setdefault("domain_label", "NOT_RELEVANT")

                graph.add_edge(source, target, **edge_attrs)

        return graph
