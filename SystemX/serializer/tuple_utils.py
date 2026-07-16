import networkx as nx
from SystemX.sca import DOMAIN_NODE_TYPES

_DATASET = DOMAIN_NODE_TYPES["DATASET"]
_MODEL = DOMAIN_NODE_TYPES["MODEL"]

def extract_tuples_from_projected(graph: nx.DiGraph) -> list[dict]:
    """Derives lineage tuples from a projected bipartite graph whose nodes are DATASET and MODEL anchors."""
    tuples: list[dict] = []
    seen: set = set()

    for u, v in graph.edges():
        if u == v:
            continue

        u_type = graph.nodes[u].get("node_type")
        v_type = graph.nodes[v].get("node_type")

        if u_type == _DATASET and v_type == _DATASET:
            key = ("<D, D>", u, v)
            if key not in seen:
                seen.add(key)
                tuples.append({"tuple_type": "<D, D>", "subject_id": u, "object_id": v})

        elif u_type == _DATASET and v_type == _MODEL:
            key = ("<M, D>", v, u)
            if key not in seen:
                seen.add(key)
                tuples.append({"tuple_type": "<M, D>", "subject_id": v, "object_id": u})

        elif u_type == _MODEL and v_type == _DATASET:
            key = ("<M, D>", u, v)
            if key not in seen:
                seen.add(key)
                tuples.append({"tuple_type": "<M, D>", "subject_id": u, "object_id": v})

    for node, attrs in graph.nodes(data=True):
        has_real_consumer = any(w != node for w in graph.successors(node))
        if attrs.get("node_type") == _DATASET and not has_real_consumer:
            key = ("<D, Empty>", node)
            if key not in seen:
                seen.add(key)
                tuples.append({"tuple_type": "<D, Empty>", "subject_id": node, "object_id": "Empty"})

    return tuples
