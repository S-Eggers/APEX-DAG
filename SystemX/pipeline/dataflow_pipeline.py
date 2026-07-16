from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline.base import Pipeline
from SystemX.serializer.dataflow_serializer import DataflowSerializer


class DataflowPipeline(Pipeline):
    def __init__(
        self,
        parser: GraphParser,
        serializer: DataflowSerializer,
        highlight_relevant: bool,
    ) -> None:
        self.parser = parser
        self.serializer = serializer
        self.highlight_relevant = highlight_relevant

    def execute(self, input_data: list) -> dict:
        dfg = self.parser.parse(input_data)

        if self.highlight_relevant:
            dfg.filter_relevant(lineage_mode=False)

        dfg.get_state().optimize()
        dfg = dfg.enrich_provenance()

        return self.serializer.to_dict(dfg)
