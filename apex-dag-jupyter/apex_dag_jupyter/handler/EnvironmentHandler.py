from .ApexDAGBaseHandler import ApexDAGBaseHandler
from ApexDAG.pipeline.environment_pipeline_factory import EnvironmentPipelineFactory

class EnvironmentHandler(ApexDAGBaseHandler):
    @property
    def response_key(self) -> str:
        return "environment_data"

    def create_pipeline(self, input_data: dict):
        return EnvironmentPipelineFactory.create(input_data)