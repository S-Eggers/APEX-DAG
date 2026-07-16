from SystemX.labeler.gat_labeler import GATLabeler
from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline.labeling_pipeline import LabelingPipeline
from SystemX.sca.refinement.factory import create_default_refiner
from SystemX.serializer.labeling_serializer import LabelingSerializer


class GATLabelingPipelineFactory:
    @staticmethod
    def create(request_payload: dict, model: dict) -> LabelingPipeline:
        use_refiner = request_payload.get("useRefiner", True)
        replace_dataflow = request_payload.get("replaceDataflowInUDFs", False)

        parser = GraphParser(replace_dataflow=replace_dataflow, enrich_provenance=request_payload.get("enrichProvenance", True), detect_dsl=request_payload.get("detectDsl", False))
        labeler = GATLabeler(model)
        refiner = create_default_refiner() if use_refiner else None
        serializer = LabelingSerializer()

        return LabelingPipeline(parser=parser, labeler=labeler, refiner=refiner, serializer=serializer)
