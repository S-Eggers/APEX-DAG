from typing import Any

from SystemX.labeler.edge_labeler import EdgeLabeler
from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline.base import Pipeline
from SystemX.sca.refinement.engine import GraphRefiner
from SystemX.serializer.lineage_serializer import LineageSerializer


class LineagePipeline(Pipeline):
    """Orchestrates the steps of lineage extraction."""

    def __init__(
        self,
        parser: GraphParser,
        labeler: EdgeLabeler,
        refiner: GraphRefiner,
        serializer: LineageSerializer,
        highlight_relevant: bool,
    ) -> None:
        self.parser = parser
        self.labeler = labeler
        self.refiner = refiner
        self.serializer = serializer
        self.highlight_relevant = highlight_relevant

    def execute(self, input_data: list) -> dict[str, Any]:
        """Runs the complete extraction pipeline."""
        dfg = self.parser.parse(input_data)
        dfg.get_state().optimize()

        self.labeler.apply_labels(dfg)

        if self.highlight_relevant:
            dfg.filter_relevant(lineage_mode=True)

        self.refiner.refine(dfg)
        dfg.get_state().optimize()

        return self.serializer.to_dict(dfg)
