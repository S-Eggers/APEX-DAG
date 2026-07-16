from SystemX.pipeline.environment_pipeline_factory import EnvironmentPipelineFactory

from .SystemXBaseHandler import SystemXBaseHandler


class EnvironmentHandler(SystemXBaseHandler):
    @property
    def response_key(self) -> str:
        return "environment_data"

    def create_pipeline(self, input_data: dict):
        return EnvironmentPipelineFactory.create(input_data)
