from ApexDAG.parser.graph_parser import GraphParser
from ApexDAG.pipeline.dataflow_pipeline import DataflowPipeline
from ApexDAG.serializer.dataflow_serializer import DataflowSerializer


class DataflowPipelineFactory:
    @staticmethod
    def create(request_payload: dict) -> DataflowPipeline:
        parser = GraphParser(replace_dataflow=request_payload.get("replaceDataflowInUDFs", False))
        serializer = DataflowSerializer()

        return DataflowPipeline(
            parser=parser,
            serializer=serializer,
            highlight_relevant=request_payload.get("highlightRelevantSubgraphs", False)
        )
