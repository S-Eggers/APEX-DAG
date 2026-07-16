from SystemX.parser.ast_parser import ASTParser
from SystemX.pipeline.ast_pipeline import ASTPipeline
from SystemX.serializer.ast_serializer import ASTSerializer


class ASTPipelineFactory:
    @staticmethod
    def create(request_payload: dict) -> ASTPipeline:
        parser = ASTParser()
        serializer = ASTSerializer()

        return ASTPipeline(parser=parser, serializer=serializer)
