import ast
from typing import Dict, Any


class EnvironmentPipeline:
    def __init__(
        self, 
        serializer: 'EnvironmentSerializer', 
        import_visitor_cls: type, 
        complexity_visitor_cls: type
    ):
        self.serializer = serializer
        self.import_visitor_cls = import_visitor_cls
        self.complexity_visitor_cls = complexity_visitor_cls

    def execute(self, code: str) -> Dict[str, Any]:
        if not code.strip():
            return self.serializer.empty_payload()

        tree = ast.parse(code)
        
        import_visitor = self.import_visitor_cls()
        complexity_visitor = self.complexity_visitor_cls()
        
        import_visitor.visit(tree)
        complexity_visitor.visit(tree)

        return self.serializer.to_dict(import_visitor, complexity_visitor)

