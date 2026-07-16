from SystemX.pipeline.execution_trace_pipeline_factory import ExecutionTracePipelineFactory

from .SystemXBaseHandler import SystemXBaseHandler


class ExecutionTraceHandler(SystemXBaseHandler):
    @property
    def response_key(self) -> str:
        return "execution_trace"

    def create_pipeline(self, input_data: dict):
        return ExecutionTracePipelineFactory.create(input_data, models=self.models)
