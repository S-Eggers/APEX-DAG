from SystemX.pipeline.ast_pipeline_factory import ASTPipelineFactory

from .SystemXBaseHandler import SystemXBaseHandler


class ASTHandler(SystemXBaseHandler):
    @property
    def response_key(self) -> str:
        return "ast_graph"

    def create_pipeline(self, input_data: dict):
        return ASTPipelineFactory.create(input_data)
