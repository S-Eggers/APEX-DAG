from SystemX.labeler.hgt_labeler import HGTLabeler
from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline.labeling_pipeline import LabelingPipeline
from SystemX.sca.refinement.factory import create_default_refiner
from SystemX.serializer.labeling_serializer import LabelingSerializer


class HGTLabelingPipelineFactory:
    @staticmethod
    def create(request_payload: dict, labeler: HGTLabeler) -> LabelingPipeline:
        use_refiner = request_payload.get("useRefiner", True)
        replace_dataflow = request_payload.get("replaceDataflowInUDFs", False)

        parser = GraphParser(replace_dataflow=replace_dataflow, enrich_provenance=request_payload.get("enrichProvenance", True), detect_dsl=request_payload.get("detectDsl", False))
        refiner = create_default_refiner() if use_refiner else None
        serializer = LabelingSerializer()

        explain = request_payload.get("explainFeatureImportance", False)

        return LabelingPipeline(parser=parser, labeler=labeler, refiner=refiner, serializer=serializer, explain=explain)
