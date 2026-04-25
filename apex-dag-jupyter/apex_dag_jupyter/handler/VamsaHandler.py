from ApexDAG.pipeline.vamsa_pipeline_factory import VamsaPipelineFactory

from .ApexDAGBaseHandler import ApexDAGBaseHandler


class VamsaHandler(ApexDAGBaseHandler):
    @property
    def response_key(self) -> str:
        return "vamsa"

    def create_pipeline(self, input_data: dict):
        return VamsaPipelineFactory.create(input_data)
