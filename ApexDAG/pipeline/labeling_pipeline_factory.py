from ApexDAG.labeler.gat_labeler import GATLabeler
from ApexDAG.labeler.llm_labeler import LLMLabeler
from ApexDAG.parser.graph_parser import GraphParser
from ApexDAG.pipeline.labeling_pipeline import LabelingPipeline
from ApexDAG.sca.graph_refiner import GraphRefiner
from ApexDAG.serializer.labeling_serializer import LabelingSerializer


class LabelingPipelineFactory:
    @staticmethod
    def create(request_payload: dict, model: dict) -> LabelingPipeline:
        use_llm = request_payload.get("llmClassification", False)
        use_refiner = request_payload.get("useRefiner", True)
        replace_dataflow = request_payload.get("replaceDataflowInUDFs", False)

        parser = GraphParser(replace_dataflow=replace_dataflow)
        labeler = LLMLabeler() if use_llm else GATLabeler(model)
        refiner = GraphRefiner() if use_refiner else None
        serializer = LabelingSerializer()

        return LabelingPipeline(
            parser=parser,
            labeler=labeler,
            refiner=refiner,
            serializer=serializer
        )
