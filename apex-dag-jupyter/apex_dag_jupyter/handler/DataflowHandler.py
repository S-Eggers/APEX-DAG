from ApexDAG.pipeline.dataflow_pipeline_factory import DataflowPipelineFactory

from .ApexDAGBaseHandler import ApexDAGBaseHandler


class DataflowHandler(ApexDAGBaseHandler):
    @property
    def response_key(self) -> str:
        return "dataflow"

    def create_pipeline(self, input_data: dict):
        return DataflowPipelineFactory.create(input_data)
