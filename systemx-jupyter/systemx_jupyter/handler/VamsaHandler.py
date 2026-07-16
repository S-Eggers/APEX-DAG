from SystemX.pipeline.vamsa_lineage_pipeline_factory import VamsaLineagePipelineFactory
from SystemX.pipeline.vamsa_pipeline_factory import VamsaPipelineFactory

from .SystemXBaseHandler import SystemXBaseHandler

class VamsaHandler(SystemXBaseHandler):
    @property
    def response_key(self) -> str:
        return "vamsa"

    def create_pipeline(self, input_data: dict):
        if input_data.get("mode", 0) == 0:
            return VamsaPipelineFactory.create(input_data)
        lineage_input = {"highlightRelevantSubgraphs": False, **input_data}
        return VamsaLineagePipelineFactory.create(lineage_input)
