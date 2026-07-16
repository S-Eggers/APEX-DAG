from SystemX.labeler.vamsa_static_labeler import VamsaStaticLabeler
from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline._shared import _load_vamsa_kb
from SystemX.pipeline.labeling_pipeline import LabelingPipeline
from SystemX.sca.refinement.factory import create_default_refiner, create_empty_refiner
from SystemX.serializer.labeling_serializer import LabelingSerializer


class VamsaLabelingPipelineFactory:
    @staticmethod
    def create(request_payload: dict) -> LabelingPipeline:
        parser = GraphParser(
            replace_dataflow=request_payload.get("replaceDataflowInUDFs", False),
            enrich_provenance=request_payload.get("enrichProvenance", True),
            detect_dsl=request_payload.get("detectDsl", False),
        )
        labeler = VamsaStaticLabeler(_load_vamsa_kb())
        refiner = create_default_refiner() if request_payload.get("useRefiner", True) else create_empty_refiner()
        return LabelingPipeline(
            parser=parser,
            labeler=labeler,
            refiner=refiner,
            serializer=LabelingSerializer(),
        )
