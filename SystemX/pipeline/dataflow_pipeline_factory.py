from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline.dataflow_pipeline import DataflowPipeline
from SystemX.serializer.dataflow_serializer import DataflowSerializer


class DataflowPipelineFactory:
    @staticmethod
    def create(request_payload: dict) -> DataflowPipeline:
        parser = GraphParser(
            replace_dataflow=request_payload.get("replaceDataflowInUDFs", False),
            enrich_provenance=request_payload.get("enrichProvenance", True),
            detect_dsl=request_payload.get("detectDsl", False),
        )
        serializer = DataflowSerializer()

        return DataflowPipeline(
            parser=parser,
            serializer=serializer,
            highlight_relevant=request_payload.get("highlightRelevantSubgraphs", False),
        )
