import json
from typing import Any

import networkx as nx
from ApexDAG.sca import NODE_TYPES, convert_multidigraph_to_digraph
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph


class DataflowSerializer:
    def to_dict(self, graph: PythonDataFlowGraph) -> dict[str, Any]:
        """
        Converts the stateful MultiDiGraph into a flattened DiGraph,
        then serializes it directly to a Cytoscape-compliant dictionary.
        """
        G = convert_multidigraph_to_digraph(graph.get_graph(), NODE_TYPES)
        json_string = self.dfg_to_json(G)

        return json.loads(json_string)

    def dfg_to_json(self, G: nx.DiGraph) -> str:
        elements = []

        for node, data in G.nodes(data=True):
            payload = {
                "id": str(node),
                "label": str(node),
                "node_type": NODE_TYPES.get("DEFAULT", 0),  # Fallback
            }
            payload.update(data)
            elements.append({"data": payload})

        for src, tgt, data in G.edges(data=True):
            payload = {
                "source": str(src),
                "target": str(tgt),
                "edge_type": 0,
                "label": data.get("code", ""),
            }
            safe_data = {k: v for k, v in data.items() if k != "id"}
            payload.update(safe_data)

            elements.append({"data": payload})

        return json.dumps({"elements": elements})
