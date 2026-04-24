import ast
from typing import Dict, Any
from ApexDAG.serializer.environment_serializer import EnvironmentSerializer

class EnvironmentPipeline:
    def __init__(
        self, 
        serializer: EnvironmentSerializer, 
        import_visitor_cls: type, 
        complexity_visitor_cls: type
    ):
        self.serializer = serializer
        self.import_visitor_cls = import_visitor_cls
        self.complexity_visitor_cls = complexity_visitor_cls

    def execute(self, cells: list) -> Dict[str, Any]:
        """
        Ingests the Jupyter cell array, parses the AST, and calculates environment metrics.
        """
        code = "\n".join([cell.get("source", "") for cell in cells])
        
        if not code.strip():
            return self.serializer.empty_payload()

        tree = ast.parse(code)
        
        import_visitor = self.import_visitor_cls()
        complexity_visitor = self.complexity_visitor_cls()
        
        import_visitor.visit(tree)
        complexity_visitor.visit(tree)

        return self.serializer.to_dict(import_visitor, complexity_visitor)