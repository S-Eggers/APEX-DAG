from ApexDAG.parser.ast_parser import ASTParser
from ApexDAG.pipeline.ast_pipeline import ASTPipeline
from ApexDAG.serializer.ast_serializer import ASTSerializer


class ASTPipelineFactory:
    @staticmethod
    def create(request_payload: dict) -> ASTPipeline:
        parser = ASTParser()
        serializer = ASTSerializer()

        return ASTPipeline(parser=parser, serializer=serializer)
