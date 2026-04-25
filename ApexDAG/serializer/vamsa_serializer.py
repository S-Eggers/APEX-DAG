from typing import Any

import networkx as nx


class VamsaSerializer:
    def to_dict(self, G: nx.DiGraph) -> dict[str, Any]:
        """
        Serializes a NetworkX DiGraph into Cytoscape.js compatible JSON.
        """
        cyto_data = nx.cytoscape_data(G)
        elements = cyto_data.get("elements", {"nodes": [], "edges": []})

        return {
            "elements": elements
        }
