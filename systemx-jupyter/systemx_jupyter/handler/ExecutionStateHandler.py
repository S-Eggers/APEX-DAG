from SystemX.pipeline.execution_state_pipeline_factory import ExecutionStatePipelineFactory

from .SystemXBaseHandler import SystemXBaseHandler


class ExecutionStateHandler(SystemXBaseHandler):
    @property
    def response_key(self) -> str:
        return "execution_state"

    def create_pipeline(self, input_data: dict):
        return ExecutionStatePipelineFactory.create(input_data, models=self.models)
