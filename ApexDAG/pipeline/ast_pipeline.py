from typing import Any

from ApexDAG.parser.ast_parser import ASTParser
from ApexDAG.serializer.ast_serializer import ASTSerializer


class ASTPipeline:
    def __init__(self, parser: ASTParser, serializer: ASTSerializer):
        self.parser = parser
        self.serializer = serializer

    def execute(self, code: str) -> dict[str, Any]:
        ast_graph = self.parser.parse(code)
        return self.serializer.to_dict(ast_graph)
