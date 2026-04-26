from typing import Any

from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph


class LabelingSerializer:
    def to_dict(self, graph: PythonDataFlowGraph) -> dict[str, Any]:
        G = graph.get_graph()
        elements = []

        for n, data in G.nodes(data=True):
            payload = {
                "id": str(n),
                "label": data.get("label") or data.get("code") or str(n),
                "node_type": 0,
            }

            payload.update(data)

            elements.append({"data": payload})

        for u, v, key, data in G.edges(keys=True, data=True):
            payload = {
                "id": f"edge_{u}_{v}_{key}",
                "source": str(u),
                "target": str(v),
                "edge_type": 2,
                "label": data.get("label") or data.get("code") or "edge",
            }

            safe_data = {k: val for k, val in data.items() if k not in ("id", "key")}
            payload.update(safe_data)

            elements.append({"data": payload})

        return {"elements": elements}
