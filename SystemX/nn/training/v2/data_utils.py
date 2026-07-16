from __future__ import annotations

import networkx as nx

from SystemX.sca.constants import canonical_domain_label

def annotation_to_networkx(elements: list) -> nx.MultiDiGraph:
    """Reconstruct a labeled MultiDiGraph from a flat Cytoscape annotation."""
    G = nx.MultiDiGraph()

    for el in elements:
        data = el.get("data", {})
        if "source" in data:
            continue
        node_id = data.get("id")
        if node_id is None:
            continue

        attrs: dict = {
            "node_type": int(data.get("node_type", 0)),
            "code": data.get("code", ""),
            "label": data.get("label", ""),
            "cell_id": data.get("cell_id", ""),
            "base_inputs": data.get("base_inputs", ""),
        }

        if "predicted_label" in data:
            import contextlib

            with contextlib.suppress(ValueError, TypeError):
                attrs["domain_label"] = canonical_domain_label(int(data["predicted_label"]))

        G.add_node(node_id, **attrs)

    for el in elements:
        data = el.get("data", {})
        if "source" not in data:
            continue
        src, tgt = data.get("source"), data.get("target")
        if src in G and tgt in G:
            G.add_edge(src, tgt, edge_type=int(data.get("edge_type", 0)))

    return G
