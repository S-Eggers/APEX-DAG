from SystemX.execution.heuristic_predictor import HeuristicOrderPredictor
from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline.execution_state_pipeline import ExecutionStatePipeline

class ExecutionStatePipelineFactory:
    @staticmethod
    def create(request_payload: dict, models: dict | None = None) -> ExecutionStatePipeline:
        parser = GraphParser(detect_dsl=request_payload.get("detectDsl", False))

        backend = request_payload.get("execBackend", "heuristic")
        predictor = HeuristicOrderPredictor()
        if backend != "heuristic" and models:
            from SystemX.execution.learned_predictor import resolve_learned_order_predictor

            predictor = resolve_learned_order_predictor(backend, models) or predictor

        return ExecutionStatePipeline(parser=parser, predictor=predictor)
