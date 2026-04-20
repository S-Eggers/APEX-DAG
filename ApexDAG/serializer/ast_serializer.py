import json
from typing import Dict, Any
from ApexDAG.sca.py_ast_graph import PythonASTGraph
from ApexDAG.util.draw import Draw

class ASTSerializer:
    def to_dict(self, graph: PythonASTGraph) -> Dict[str, Any]:
        draw = Draw(None, None)
        G = graph.get_graph()

        json_string = draw.ast_to_json(G)
        return json.loads(json_string)