import ast

from SystemX.execution.heuristic_predictor import HeuristicOrderPredictor
from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline.base import Pipeline
from SystemX.sca.cell_graph_projector import CellGraphProjector

class ExecutionStatePipeline(Pipeline):
    """GraphParser -> CellGraphProjector -> order predictor -> JSON report."""

    def __init__(self, parser: GraphParser, predictor: HeuristicOrderPredictor) -> None:
        self.parser = parser
        self.predictor = predictor

    def execute(self, input_data: list) -> dict:
        parseable = [cell for cell in input_data if self._parses(str(cell.get("source", "")))]
        dfg = self.parser.parse(parseable)

        projector = CellGraphProjector(dfg, input_data)
        cell_graph = projector.project()
        report = self.predictor.predict(cell_graph, dfg=dfg)

        result = report.to_dict()
        result["cell_graph"] = cell_graph.to_serializable()
        return result

    @staticmethod
    def _parses(source: str) -> bool:
        sanitized = CellGraphProjector._sanitize(source)
        try:
            ast.parse(sanitized)
        except SyntaxError:
            return False
        return True
