from ApexDAG.parser.graph_parser import GraphParser
from ApexDAG.labeler.llm_labeler import LLMLabeler
from ApexDAG.labeler.gat_labeler import GATLabeler
from ApexDAG.sca.graph_refiner import GraphRefiner
from ApexDAG.serializer.lineage_serializer import LineageSerializer
from ApexDAG.pipeline.lineage_pipeline import LineagePipeline


class LineagePipelineFactory:
    @staticmethod
    def create(request_payload: dict, model: dict) -> LineagePipeline:
        use_llm = request_payload.get("llmClassification", False)
        
        parser = GraphParser(replace_dataflow=request_payload.get("replaceDataflowInUDFs", False))
        labeler = LLMLabeler() if use_llm else GATLabeler(model)
        refiner = GraphRefiner()
        serializer = LineageSerializer()
        
        return LineagePipeline(
            parser=parser,
            labeler=labeler,
            refiner=refiner,
            serializer=serializer,
            highlight_relevant=request_payload.get("highlightRelevantSubgraphs", True)
        )