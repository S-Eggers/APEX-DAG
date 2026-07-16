from SystemX.labeler.xgboost_labeler import XGBoostLabeler
from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline.lineage_pipeline import LineagePipeline
from SystemX.sca.refinement.factory import create_default_refiner
from SystemX.serializer.lineage_serializer import LineageSerializer


class XGBoostLineagePipelineFactory:
    @staticmethod
    def create(request_payload: dict, labeler: XGBoostLabeler) -> LineagePipeline:
        parser = GraphParser(
            replace_dataflow=request_payload.get("replaceDataflowInUDFs", False),
            enrich_provenance=request_payload.get("enrichProvenance", True),
            detect_dsl=request_payload.get("detectDsl", False),
        )
        refiner = create_default_refiner()
        serializer = LineageSerializer()

        return LineagePipeline(
            parser=parser,
            labeler=labeler,
            refiner=refiner,
            serializer=serializer,
            highlight_relevant=request_payload.get("highlightRelevantSubgraphs", True),
        )
