from .ApexDAGBaseHandler import ApexDAGBaseHandler
from ApexDAG.pipeline.lineage_pipeline_factory import LineagePipelineFactory

class LineageHandler(ApexDAGBaseHandler):
    @property
    def response_key(self) -> str:
        return "lineage_predictions"

    def create_pipeline(self, input_data: dict):
        return LineagePipelineFactory.create(input_data, self.model)