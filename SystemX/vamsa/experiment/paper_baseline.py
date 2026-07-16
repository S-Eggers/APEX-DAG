import ast
import logging
import re
from pathlib import Path

from SystemX.labeling.extractor import NotebookExtractor
from SystemX.mining.extract_tuples.domain import LineageTuple
from SystemX.vamsa.core import reset_vamsa_counter
from SystemX.vamsa.extraction import gen_wir
from SystemX.vamsa.lineage import KB, AnnotationWIR
from SystemX.vamsa.lineage.vamsa_mapper import VamsaTupleMapper

from .models import TextSpan
from .spatial_node_matcher import SpatialNodeMatcher

logger = logging.getLogger(__name__)

_MAGIC_LINE = re.compile(r"^\s*(%|!|\?)")

Tuple = tuple[str, str, str]

def _sanitize(source: str) -> str:
    """Comment out IPython magics/shell lines so ast.parse succeeds."""
    lines = source.split("\n")
    return "\n".join("# <stripped magic>" if _MAGIC_LINE.match(line) else line for line in lines)

def _join_cells(cells: list[dict]) -> tuple[str, dict[str, int]]:
    """Concatenate cell sources; returns (code, cell_id -> 0-based line offset)."""
    offsets: dict[str, int] = {}
    parts: list[str] = []
    cursor = 0
    for cell in cells:
        src = _sanitize(cell["source"])
        offsets[cell["cell_id"]] = cursor
        parts.append(src)
        cursor += src.count("\n") + 1
    return "\n".join(parts), offsets

def _int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1

def extract_paper_tuples(
    golden_elements: list[dict],
    notebook_path: Path,
    kb_csv_path: str | None = None,
) -> set[Tuple]:
    """Vamsa lineage tuples for one notebook, in golden-graph node-id space."""
    cells = NotebookExtractor.to_structured_cells(notebook_path)
    if not cells:
        raise ValueError(f"No code cells extracted from {notebook_path}")

    code, cell_offsets = _join_cells(cells)
    tree = ast.parse(code)

    reset_vamsa_counter()
    wir_graph, prs, _tuples = gen_wir(tree)
    annotator = AnnotationWIR(wir_graph, prs, KB(kb_csv_path=kb_csv_path))
    annotated = annotator.annotate()

    raw_tuples: list[LineageTuple] = VamsaTupleMapper(annotated).extract()
    if not raw_tuples:
        return set()

    matcher = SpatialNodeMatcher({"elements": golden_elements}, cell_offsets)

    label_to_golden: dict[str, str] = {}
    for element in golden_elements:
        data = element.get("data", {})
        if "id" in data and "source" not in data:
            label = str(data.get("label", ""))
            if label and label not in label_to_golden:
                label_to_golden[label] = str(data["id"])

    mapping_cache: dict[str, str] = {}

    def to_golden(node_id: str) -> str:
        if node_id == "Empty":
            return "Empty"
        if node_id not in mapping_cache:
            data = annotated.nodes[node_id] if node_id in annotated.nodes else {}
            span = TextSpan(
                start_line=_int(data.get("lineno")),
                start_col=_int(data.get("col_offset")),
                end_line=_int(data.get("end_lineno")),
                end_col=_int(data.get("end_col_offset")),
            )
            match = matcher.find_best_match(str(node_id), span)
            if not match:
                match = label_to_golden.get(str(node_id).split(":")[0])
            mapping_cache[node_id] = match if match else f"UNMAPPED::{node_id}"
        return mapping_cache[node_id]

    return {(t.tuple_type, to_golden(str(t.subject_id)), to_golden(str(t.object_id))) for t in raw_tuples}
