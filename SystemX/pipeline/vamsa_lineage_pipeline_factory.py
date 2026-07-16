from SystemX.labeler.vamsa_static_labeler import VamsaStaticLabeler
from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline._shared import _load_vamsa_kb
from SystemX.pipeline.lineage_pipeline import LineagePipeline
from SystemX.sca.refinement.factory import create_default_refiner
from SystemX.serializer.lineage_serializer import LineageSerializer


class VamsaLineagePipelineFactory:
    @staticmethod
    def create(request_payload: dict) -> LineagePipeline:
        parser = GraphParser(
            replace_dataflow=request_payload.get("replaceDataflowInUDFs", False),
            enrich_provenance=request_payload.get("enrichProvenance", True),
            detect_dsl=request_payload.get("detectDsl", False),
        )
        labeler = VamsaStaticLabeler(_load_vamsa_kb())
        return LineagePipeline(
            parser=parser,
            labeler=labeler,
            refiner=create_default_refiner(),
            serializer=LineageSerializer(),
            highlight_relevant=request_payload.get("highlightRelevantSubgraphs", True),
        )
