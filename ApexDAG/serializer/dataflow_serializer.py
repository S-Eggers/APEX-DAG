import json
from typing import Dict, Any
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph
from ApexDAG.util.draw import Draw
from ApexDAG.sca import NODE_TYPES, EDGE_TYPES, convert_multidigraph_to_digraph

class DataflowSerializer:
    def to_dict(self, graph: PythonDataFlowGraph) -> Dict[str, Any]:
        draw = Draw(NODE_TYPES, EDGE_TYPES)
        G = convert_multidigraph_to_digraph(graph.get_graph(), NODE_TYPES)
        json_string = draw.dfg_to_json(G)

        return json.loads(json_string)