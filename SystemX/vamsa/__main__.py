import argparse
import ast
import json
import logging
import warnings
from pathlib import Path

import nbformat

from SystemX.experiment.evaluation.vamsa_evaluator import EdgeLabeled, EdgeStruct, VamsaSemanticEvaluator
from SystemX.parser.sanitizer_mixin import IPythonSanitizerMixin
from SystemX.util.logger import configure_systemx_logger
from SystemX.vamsa import execute_vamsa_pipeline
from SystemX.vamsa.experiment.golden_types import GoldenElementData, GoldenGraph
from SystemX.vamsa.experiment.graph_projector import GraphProjector
from SystemX.vamsa.experiment.spatial_node_matcher import SpatialNodeMatcher

configure_systemx_logger()
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=SyntaxWarning, message=".*invalid.*")


class NotebookLoader(IPythonSanitizerMixin):
    def load_and_sanitize_with_offsets(self, ipynb_path: Path) -> tuple[str, dict[str, int]]:
        with open(ipynb_path, encoding="utf-8") as f:
            notebook_data = nbformat.read(f, as_version=4)

        clean_lines: list[str] = []
        cell_offsets: dict[str, int] = {}
        current_line = 1

        code_cells = [c for c in notebook_data.cells if c.cell_type == "code"]
        sanitized_cells = self.sanitize_ipython_cells(code_cells)

        for fallback_counter, cell in enumerate(sanitized_cells):
            cell_id = f"fallback-{fallback_counter}"
            cell_offsets[cell_id] = current_line - 1

            source_text = str(cell["source"])
            lines = source_text.split("\n")

            clean_lines.extend(lines)
            clean_lines.append("\n")
            current_line += len(lines) + 1

        return "\n".join(clean_lines), cell_offsets


def extract_golden_edges_labeled(golden_json: GoldenGraph) -> set[EdgeLabeled]:
    """Extracts edges alongside their semantic label."""
    edges: set[EdgeLabeled] = set()
    for element in golden_json.get("elements", []):
        data: GoldenElementData = element.get("data", {})
        source = data.get("source")
        target = data.get("target")

        label = data.get("predicted_label") if "predicted_label" in data else data.get("edge_type")

        if source is not None and target is not None and label is not None:
            edges.add((str(source), str(target), int(label)))
    return edges


def execute_evaluation_pipeline(raw_dir: Path, annotations_dir: Path, kb_csv_path: str) -> None:
    if not raw_dir.exists() or not annotations_dir.exists():
        logger.error("Provided data directories do not exist.")
        return

    loader = NotebookLoader()
    evaluator = VamsaSemanticEvaluator()

    for json_path in annotations_dir.glob("*.json"):
        notebook_name = json_path.name.replace(".json", ".ipynb")
        ipynb_path = raw_dir / notebook_name

        if not ipynb_path.exists():
            continue

        logger.info(f"Evaluating: {notebook_name}")

        with open(json_path, encoding="utf-8") as f:
            golden_json: GoldenGraph = json.load(f)

        golden_labeled = extract_golden_edges_labeled(golden_json)
        golden_struct = {(u, v) for u, v, _ in golden_labeled}

        source_code, cell_offsets = loader.load_and_sanitize_with_offsets(ipynb_path)

        try:
            ast_tree = ast.parse(source_code)
            vamsa_graph, _, _ = execute_vamsa_pipeline(ast_tree, what_track={"features"}, kb_csv_path=kb_csv_path)

            matcher = SpatialNodeMatcher(golden_json, cell_offsets)
            projector = GraphProjector(vamsa_graph, matcher)

            golden_struct_to_label: dict[EdgeStruct, int] = {(u, v): label for u, v, label in golden_labeled}

            pred_struct: set[EdgeStruct] = projector.project_edges()

            pred_labeled: set[EdgeLabeled] = {(u, v, golden_struct_to_label.get((u, v), -1)) for u, v in pred_struct}

            evaluator.record_success(vamsa_nodes_count=vamsa_graph.number_of_nodes(), mapped_nodes_count=len(projector.node_mapping))
            evaluator.update(pred_labeled, golden_labeled, pred_struct, golden_struct)

        except Exception as e:
            logger.error(f"Vamsa parsing/execution failed for {notebook_name}: {e}")
            evaluator.record_failure(golden_labeled)

    evaluator.report()


def main() -> None:
    parser = argparse.ArgumentParser(description="Vamsa Semantic Baseline Engine")
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--annotations_dir", type=str, required=True)
    parser.add_argument("--kb_csv", type=str, required=True)
    args = parser.parse_args()

    execute_evaluation_pipeline(Path(args.raw_dir), Path(args.annotations_dir), args.kb_csv)


if __name__ == "__main__":
    main()
