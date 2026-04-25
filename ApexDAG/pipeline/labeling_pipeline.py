from typing import Any

from ApexDAG.labeler.edge_labeler import EdgeLabeler
from ApexDAG.parser.graph_parser import GraphParser
from ApexDAG.sca.graph_refiner import GraphRefiner
from ApexDAG.serializer.labeling_serializer import LabelingSerializer


class LabelingPipeline:
    def __init__(self, parser: GraphParser, labeler: EdgeLabeler, refiner: GraphRefiner, serializer: LabelingSerializer):
        self.parser = parser
        self.labeler = labeler
        self.refiner = refiner
        self.serializer = serializer

    def execute(self, code: str) -> dict[str, Any]:
        dfg = self.parser.parse(code)
        dfg.get_state().optimize()

        self.labeler.apply_labels(dfg)

        if self.refiner:
            self.refiner.refine(dfg)

        return self.serializer.to_dict(dfg)
