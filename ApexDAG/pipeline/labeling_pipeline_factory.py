from ApexDAG.parser.graph_parser import GraphParser
from ApexDAG.labeler.llm_labeler import LLMLabeler
from ApexDAG.labeler.gat_labeler import GATLabeler
from ApexDAG.sca.graph_refiner import GraphRefiner
from ApexDAG.serializer.labeling_serializer import LabelingSerializer
from ApexDAG.pipeline.labeling_pipeline import LabelingPipeline


class LabelingPipelineFactory:
    @staticmethod
    def create(request_payload: dict, model: dict) -> LabelingPipeline:
        use_llm = request_payload.get("llmClassification", False)
        parser = GraphParser(replace_dataflow=request_payload.get("replaceDataflowInUDFs", False))
        
        labeler = LLMLabeler() if use_llm else GATLabeler(model)
        refiner = GraphRefiner()
        serializer = LabelingSerializer()
        
        return LabelingPipeline(
            parser=parser, 
            labeler=labeler, 
            refiner=refiner, 
            serializer=serializer
        )