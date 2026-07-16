from SystemX.llm.config import Config
from SystemX.pipeline._shared import _build_llm_components
from SystemX.pipeline.labeling_pipeline import LabelingPipeline
from SystemX.serializer.labeling_serializer import LabelingSerializer


class LLMLabelingPipelineFactory:
    @staticmethod
    def create(config: Config) -> LabelingPipeline:
        parser, labeler, refiner = _build_llm_components(config)
        return LabelingPipeline(
            parser=parser,
            labeler=labeler,
            refiner=refiner,
            serializer=LabelingSerializer(),
        )
