from .ApexDAGBaseHandler import ApexDAGBaseHandler
from ApexDAG.pipeline.ast_pipeline_factory import ASTPipelineFactory

class ASTHandler(ApexDAGBaseHandler):
    @property
    def response_key(self) -> str:
        return "ast_graph"

    def create_pipeline(self, input_data: dict):
        return ASTPipelineFactory.create(input_data)