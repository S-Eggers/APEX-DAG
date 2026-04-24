from .ApexDAGBaseHandler import ApexDAGBaseHandler
from ApexDAG.pipeline.vamsa_pipeline_factory import VamsaPipelineFactory

class VamsaHandler(ApexDAGBaseHandler):
    @property
    def response_key(self) -> str:
        return "vamsa"

    def create_pipeline(self, input_data: dict):
        return VamsaPipelineFactory.create(input_data)