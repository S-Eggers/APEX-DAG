from SystemX.pipeline.dataflow_pipeline_factory import DataflowPipelineFactory

from .SystemXBaseHandler import SystemXBaseHandler


class DataflowHandler(SystemXBaseHandler):
    @property
    def response_key(self) -> str:
        return "dataflow"

    def create_pipeline(self, input_data: dict):
        return DataflowPipelineFactory.create(input_data)
