from typing import Any

from ApexDAG.parser.graph_parser import GraphParser
from ApexDAG.serializer.dataflow_serializer import DataflowSerializer


class DataflowPipeline:
    def __init__(
        self,
        parser: GraphParser,
        serializer: DataflowSerializer,
        highlight_relevant: bool
    ):
        self.parser = parser
        self.serializer = serializer
        self.highlight_relevant = highlight_relevant

    def execute(self, code: str) -> dict[str, Any]:
        dfg = self.parser.parse(code)

        if self.highlight_relevant:
            dfg.filter_relevant(lineage_mode=False)

        dfg.get_state().optimize()

        return self.serializer.to_dict(dfg)
