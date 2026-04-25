import json
import os

import networkx as nx


class Draw:
    def __init__(self, node_types: dict, edge_types: dict) -> None:
        self.NODE_TYPES = node_types
        self.EDGE_TYPES = edge_types

    def dfg_webrendering(self, G: nx.DiGraph, save_path: str = None):
        file_name = os.path.basename(save_path) if save_path else "data_flow_graph"
        directory_name = os.path.dirname(save_path) if save_path else "output"
        directory = os.path.join(os.getcwd(), directory_name)

        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Save JSON
        with open(os.path.join(directory, f"{file_name}.json"), "w") as f:
            f.write(self.dfg_to_json(G))

    def dfg_to_json(self, G: nx.DiGraph) -> str:
        elements = []
        # Convert nodes
        for node, data in G.nodes(data=True):
            elements.append(
                {
                    "data": {
                        "id": str(node),
                        "label": data.get("label", str(node)),
                        "node_type": data.get("node_type", "default"),
                    }
                }
            )

        # Convert edges
        for src, tgt, data in G.edges(data=True):
            elements.append(
                {
                    "data": {
                        "source": str(src),
                        "target": str(tgt),
                        "edge_type": data.get("edge_type", "default"),
                        "label": data.get("code", ""),  # Edge label
                        "predicted_label": data.get(
                            "predicted_label", ""
                        ),  # Edge label
                    }
                }
            )

        return json.dumps({"elements": elements})

    def ast_to_json(self, G: nx.DiGraph) -> str:
        """
        Serializes the NetworkX DiGraph into a Cytoscape-compatible JSON payload.
        Dynamically unpacks all node and edge attributes to prevent data loss.
        """
        import json
        elements = []

        for node, data in G.nodes(data=True):
            payload = {
                "id": str(node),
                "node_type": 0,
                "label": str(node)
            }

            payload.update(data)

            elements.append({"data": payload})

        # Convert edges
        for src, tgt, data in G.edges(data=True):
            payload = {
                "source": str(src),
                "target": str(tgt),
                "edge_type": 0,
                "label": "edge"
            }

            safe_data = {k: v for k, v in data.items() if k != "id"}
            payload.update(safe_data)

            elements.append({"data": payload})

        return json.dumps({"elements": elements})
