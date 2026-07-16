from SystemX.llm.config import Config
from SystemX.pipeline._shared import _build_llm_components
from SystemX.pipeline.lineage_pipeline import LineagePipeline
from SystemX.serializer.lineage_serializer import LineageSerializer


class LLMLineagePipelineFactory:
    @staticmethod
    def create(request_payload: dict, config: Config) -> LineagePipeline:
        parser, labeler, refiner = _build_llm_components(config)
        return LineagePipeline(
            parser=parser,
            labeler=labeler,
            refiner=refiner,
            serializer=LineageSerializer(),
            highlight_relevant=request_payload.get("highlightRelevantSubgraphs", True),
        )
