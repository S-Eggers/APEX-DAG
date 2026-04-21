from typing import Dict, Any
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph

class LabelingSerializer:
    def to_dict(self, graph: PythonDataFlowGraph) -> Dict[str, Any]:
        G = graph.get_graph()
        elements = []
        
        for n, data in G.nodes(data=True):
            node_label = data.get("label") or data.get("code") or str(n)
            elements.append({
                "data": {
                    "id": str(n),
                    "label": node_label,
                    "node_type": data.get("node_type", 0),
                    "code": data.get("code", ""),
                    "predicted_label": data.get("predicted_label", ""),
                    "domain_label": data.get("domain_label", "")
                }
            })
            
        for u, v, key, data in G.edges(keys=True, data=True):
            edge_str = data.get("label") or data.get("code") or "edge"
            elements.append({
                "data": {
                    "source": str(u),
                    "target": str(v),
                    "edge_type": data.get("edge_type", 2),
                    "label": edge_str,
                    "predicted_label": data.get("predicted_label", 2),
                    "domain_label": data.get("domain_label", "")
                }
            })
            
        return {"elements": elements}