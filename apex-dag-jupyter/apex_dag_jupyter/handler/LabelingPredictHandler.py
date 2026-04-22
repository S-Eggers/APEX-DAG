from .ApexDAGBaseHandler import ApexDAGBaseHandler
from ApexDAG.pipeline.labeling_pipeline_factory import LabelingPipelineFactory

class LabelingPredictHandler(ApexDAGBaseHandler):
    @property
    def response_key(self) -> str:
        return "predictions"

    def create_pipeline(self, input_data: dict):
        return LabelingPipelineFactory.create(input_data, self.model)