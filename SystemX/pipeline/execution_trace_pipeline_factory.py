from SystemX.execution.heuristic_predictor import HeuristicOrderPredictor
from SystemX.parser.graph_parser import GraphParser
from SystemX.pipeline.execution_trace_pipeline import ExecutionTracePipeline


class ExecutionTracePipelineFactory:
    @staticmethod
    def create(request_payload: dict, models: dict | None = None) -> ExecutionTracePipeline:
        parser = GraphParser(detect_dsl=request_payload.get("detectDsl", False))

        backend = request_payload.get("execBackend", "heuristic")
        predictor = HeuristicOrderPredictor()
        if backend != "heuristic" and models:
            from SystemX.execution.learned_predictor import resolve_learned_order_predictor

            predictor = resolve_learned_order_predictor(backend, models) or predictor

        return ExecutionTracePipeline(
            parser=parser,
            predictor=predictor,
            trace_payload=request_payload.get("trace"),
        )
