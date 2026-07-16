import json

import networkx as nx
from SystemX.sca import NODE_TYPES, convert_multidigraph_to_digraph
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation
from SystemX.serializer.base import CytoscapeSerializerMixin

class DataflowSerializer(CytoscapeSerializerMixin):
    def to_dict(self, graph: CDFIntermediateRepresentation) -> dict:
        """Converts the stateful MultiDiGraph into a flattened DiGraph, enriches it with provenance history, and serializes it."""
        G = convert_multidigraph_to_digraph(graph.get_graph(), NODE_TYPES)
        json_string = self.dfg_to_json(G)
        return json.loads(json_string)

    def dfg_to_json(self, graph: nx.DiGraph) -> str:
        elements = []

        for node, data in graph.nodes(data=True):
            elements.append(
                self._node_element(
                    node,
                    data,
                    {"label": str(node), "node_type": NODE_TYPES.get("DEFAULT", 0)},
                )
            )

        for src, tgt, data in graph.edges(data=True):
            elements.append(
                self._edge_element(
                    src,
                    tgt,
                    data,
                    {"edge_type": 0, "label": data.get("code", "")},
                )
            )

        return json.dumps({"elements": elements})
