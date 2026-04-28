import re
import time
import traceback
from dataclasses import dataclass
from typing import Any

from ApexDAG.pipeline.dataflow_pipeline_factory import DataflowPipelineFactory


@dataclass
class ValidationMetrics:
    success: bool
    edge_count: int = 0
    node_count: int = 0
    lines_of_code: int = 0
    contains_ml_semantics: bool = False
    extraction_time_sec: float = -1.0
    error_type: str | None = None
    stacktrace: str | None = None


class PipelineValidator:
    """Tests the notebook against the Dataflow Pipeline without saving the graph."""

    def __init__(self) -> None:
        self.pipeline = DataflowPipelineFactory.create(
            {"replaceDataflowInUDFs": False, "highlightRelevantSubgraphs": False}
        )

        self.ml_pattern = re.compile(
            r"\b(sklearn|torch|tensorflow|keras|xgboost|lightgbm|catboost|tensor|polars|pyspark)\b|"
            r"\.fit\(|\.predict\(|\.evaluate\(|\.backward\(|\.transpose\(|\.reshape\("
        )

    def validate(self, notebook_dict: dict[str, Any]) -> ValidationMetrics:
        start_time = time.time()
        api_cells = []
        loc = 0
        has_ml_semantics = False

        for i, cell in enumerate(notebook_dict.get("cells", [])):
            if cell.get("cell_type") == "code":
                raw_source = cell.get("source")

                if isinstance(raw_source, list):
                    source = "".join(raw_source)
                elif isinstance(raw_source, str):
                    source = raw_source
                else:
                    source = ""

                if source.strip():
                    api_cells.append(
                        {"cell_id": cell.get("id", f"cell_{i}"), "source": source}
                    )
                    loc += len(source.splitlines())

                    if not has_ml_semantics and self.ml_pattern.search(source):
                        has_ml_semantics = True

        if not api_cells:
            return ValidationMetrics(success=False, error_type="EmptyNotebook")

        try:
            analysis_results = self.pipeline.execute(api_cells)
            elements = (
                analysis_results.get("elements", [])
                if isinstance(analysis_results, dict)
                else analysis_results
            )

            if not isinstance(elements, list):
                message = f"""
                Pipeline did not return a valid elements list.
                Got: {type(elements)}
                """
                raise ValueError(message)

            edges = sum(1 for el in elements if "source" in el.get("data", {}))
            nodes = sum(
                1
                for el in elements
                if "source" not in el.get("data", {}) and "id" in el.get("data", {})
            )

            return ValidationMetrics(
                success=True,
                edge_count=edges,
                node_count=nodes,
                lines_of_code=loc,
                contains_ml_semantics=has_ml_semantics,
                extraction_time_sec=time.time() - start_time,
            )

        except Exception as e:
            return ValidationMetrics(
                success=False,
                lines_of_code=loc,
                contains_ml_semantics=has_ml_semantics,
                extraction_time_sec=time.time() - start_time,
                error_type=e.__class__.__name__,
                stacktrace=traceback.format_exc(),
            )
