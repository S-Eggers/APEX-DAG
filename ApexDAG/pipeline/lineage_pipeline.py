from typing import Dict, Any
from ApexDAG.parser.graph_parser import GraphParser
from ApexDAG.sca.graph_refiner import GraphRefiner
from ApexDAG.labeler.edge_labeler import EdgeLabeler
from ApexDAG.serializer.lineage_serializer import LineageSerializer


class LineagePipeline:
    """Orchestrates the steps of lineage extraction."""
    
    def __init__(
        self,
        parser: GraphParser,
        labeler: EdgeLabeler,
        refiner: GraphRefiner,
        serializer: LineageSerializer,
        highlight_relevant: bool
    ):
        self.parser = parser
        self.labeler = labeler
        self.refiner = refiner
        self.serializer = serializer
        self.highlight_relevant = highlight_relevant

    def execute(self, code: str) -> Dict[str, Any]:
        """Runs the complete extraction pipeline."""
        dfg = self.parser.parse(code)
        dfg.get_state().optimize()
        
        self.labeler.apply_labels(dfg)
        
        if self.highlight_relevant:
            dfg.filter_relevant(lineage_mode=True)
        
        self.refiner.refine(dfg)
        dfg.get_state().optimize()

        return self.serializer.to_dict(dfg)