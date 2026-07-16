import inspect
import logging
from typing import Any

from SystemX.labeler.edge_labeler import EdgeLabeler
from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline.base import Pipeline
from SystemX.sca.refinement.engine import GraphRefiner
from SystemX.serializer.labeling_serializer import LabelingSerializer

logger = logging.getLogger(__name__)

class LabelingPipeline(Pipeline):
    def __init__(
        self,
        parser: GraphParser,
        labeler: EdgeLabeler,
        refiner: GraphRefiner,
        serializer: LabelingSerializer,
        explain: bool = False,
    ) -> None:
        self.parser = parser
        self.labeler = labeler
        self.refiner = refiner
        self.serializer = serializer
        self.explain = explain

    def execute(self, input_data: list) -> dict[str, Any]:
        dfg = self.parser.parse(input_data)
        dfg.get_state().optimize()

        self._apply_labels(dfg)

        if self.refiner:
            self.refiner.refine(dfg)

        dfg.enrich_provenance()

        return self.serializer.to_dict(dfg)

    def _apply_labels(self, dfg: object) -> None:
        """Label nodes, forwarding explain only to labelers that accept it."""
        if self.explain and "explain" in inspect.signature(self.labeler.apply_labels).parameters:
            self.labeler.apply_labels(dfg, explain=True)
            return
        if self.explain:
            logger.info("Labeler %s does not support feature-importance; skipping.", type(self.labeler).__name__)
        self.labeler.apply_labels(dfg)
