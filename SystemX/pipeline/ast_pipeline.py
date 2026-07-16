from typing import Any

from SystemX.parser.ast_parser import ASTParser
from SystemX.pipeline.base import Pipeline
from SystemX.serializer.ast_serializer import ASTSerializer


class ASTPipeline(Pipeline):
    def __init__(self, parser: ASTParser, serializer: ASTSerializer) -> None:
        self.parser = parser
        self.serializer = serializer

    def execute(self, input_data: list) -> dict[str, Any]:
        ast_graph = self.parser.parse(input_data)
        return self.serializer.to_dict(ast_graph)
