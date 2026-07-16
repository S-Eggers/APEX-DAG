import ast

from SystemX.execution.heuristic_predictor import HeuristicOrderPredictor
from SystemX.execution.trace import (
    analyze_session,
    freshness_map,
    minimal_replay_set,
    parse_trace,
    reproducibility_report,
)
from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline.base import Pipeline
from SystemX.sca.cell_graph_projector import CellGraphProjector

class ExecutionTracePipeline(Pipeline):
    """GraphParser -> CellGraphProjector -> predictor -> trace replay."""

    def __init__(self, parser: GraphParser, predictor: HeuristicOrderPredictor, trace_payload: dict | None) -> None:
        self.parser = parser
        self.predictor = predictor
        self.trace_payload = trace_payload or {}

    def execute(self, input_data: list) -> dict:
        parseable = [cell for cell in input_data if self._parses(str(cell.get("source", "")))]
        dfg = self.parser.parse(parseable)

        cell_graph = CellGraphProjector(dfg, input_data).project()
        report = self.predictor.predict(cell_graph, dfg=dfg)
        predicted_order = report.predicted_order

        sessions, sources = parse_trace(self.trace_payload)
        current_cell_ids = set(cell_graph.graph.nodes)
        analyses = [
            analyze_session(cell_graph, session, current_cell_ids, sources, predicted_order)
            for session in sessions
        ]

        current_session = sessions[-1] if sessions else None
        freshness = freshness_map(cell_graph, current_session, input_data)
        replay_sets = {cell: minimal_replay_set(cell_graph, freshness, cell) for cell in cell_graph.cells}
        reproducibility = reproducibility_report(cell_graph, analyses[-1] if analyses else None, freshness)

        return {
            "predicted_order": predicted_order,
            "cell_graph": cell_graph.to_serializable(),
            "sessions": analyses,
            "current_session": analyses[-1] if analyses else None,
            "freshness": freshness,
            "replay_sets": replay_sets,
            "reproducibility": reproducibility,
        }

    @staticmethod
    def _parses(source: str) -> bool:
        sanitized = CellGraphProjector._sanitize(source)
        try:
            ast.parse(sanitized)
        except SyntaxError:
            return False
        return True
