from typing import Any

from SystemX.sca.cdf_ir import CDFIntermediateRepresentation
from SystemX.serializer.base import CytoscapeSerializerMixin


class LabelingSerializer(CytoscapeSerializerMixin):
    def to_dict(self, graph: CDFIntermediateRepresentation) -> dict[str, Any]:
        G = graph.get_graph()
        elements = []

        for n, data in G.nodes(data=True):
            elements.append(
                self._node_element(
                    n,
                    data,
                    {
                        "label": data.get("label") or data.get("code") or str(n),
                        "node_type": 0,
                    },
                )
            )

        for u, v, key, data in G.edges(keys=True, data=True):
            elements.append(
                self._edge_element(
                    u,
                    v,
                    data,
                    {
                        "id": f"edge_{u}_{v}_{key}",
                        "edge_type": 2,
                        "label": data.get("label") or data.get("code") or "edge",
                    },
                    exclude=frozenset({"id", "key"}),
                )
            )

        return {"elements": elements}
