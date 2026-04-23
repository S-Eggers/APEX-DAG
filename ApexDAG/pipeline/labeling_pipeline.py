from typing import Dict, Any
from ApexDAG.parser.graph_parser import GraphParser
from ApexDAG.labeler.edge_labeler import EdgeLabeler

class LabelingPipeline:
    def __init__(self, parser: GraphParser, labeler: EdgeLabeler, serializer: 'LabelingSerializer'):
        self.parser = parser
        self.labeler = labeler
        self.serializer = serializer

    def execute(self, code: str) -> Dict[str, Any]:
        dfg = self.parser.parse(code)
        dfg.get_state().optimize()
        
        self.labeler.apply_labels(dfg)
        
        return self.serializer.to_dict(dfg)