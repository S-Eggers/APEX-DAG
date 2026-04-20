from ApexDAG.parser.ast_parser import ASTParser
from ApexDAG.serializer.ast_serializer import ASTSerializer
from typing import Dict, Any


class ASTPipeline:
    def __init__(self, parser: ASTParser, serializer: ASTSerializer):
        self.parser = parser
        self.serializer = serializer

    def execute(self, code: str) -> Dict[str, Any]:
        ast_graph = self.parser.parse(code)
        return self.serializer.to_dict(ast_graph)