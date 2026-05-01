import json
from typing import Any

import networkx as nx
from ApexDAG.sca.py_ast_graph import PythonASTGraph


class ASTSerializer:
    def to_dict(self, graph: PythonASTGraph) -> dict[str, Any]:
        ast = graph.get_graph()

        json_string = self.ast_to_json(ast)
        return json.loads(json_string)

    def ast_to_json(self, graph: nx.DiGraph) -> str:
        """
        Serializes the NetworkX DiGraph into a Cytoscape-compatible JSON payload.
        Dynamically unpacks all node and edge attributes to prevent data loss.
        """
        elements = []

        for node, data in graph.nodes(data=True):
            payload = {"id": str(node), "node_type": 0, "label": str(node)}

            payload.update(data)

            elements.append({"data": payload})

        # Convert edges
        for src, tgt, data in graph.edges(data=True):
            payload = {
                "source": str(src),
                "target": str(tgt),
                "edge_type": 0,
                "label": "edge",
            }

            safe_data = {k: v for k, v in data.items() if k != "id"}
            payload.update(safe_data)

            elements.append({"data": payload})

        return json.dumps({"elements": elements})
