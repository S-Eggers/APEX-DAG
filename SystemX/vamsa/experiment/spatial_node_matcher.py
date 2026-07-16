import logging

from .golden_types import GoldenElementData, GoldenGraph
from .models import TextSpan

logger = logging.getLogger(__name__)

class SpatialNodeMatcher:
    """Aligns nodes between two graphs using edge-derived spatial heuristics and global offsets."""

    def __init__(self, golden_graph_json: GoldenGraph, cell_offsets: dict[str, int]) -> None:
        self.golden_nodes_spatial = self._parse_golden_nodes(golden_graph_json, cell_offsets)
        self.golden_nodes_raw = self._parse_golden_strings(golden_graph_json)

    @staticmethod
    def _as_int(value: object) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _parse_golden_nodes(self, json_data: GoldenGraph, cell_offsets: dict[str, int]) -> dict[str, TextSpan]:
        nodes: dict[str, TextSpan] = {}
        for element in json_data.get("elements", []):
            data: GoldenElementData = element.get("data", {})

            if "source" in data and "target" in data and "lineno" in data:
                cell_id = data.get("cell_id", "")
                offset = cell_offsets.get(cell_id, 0)

                start_line = self._as_int(data.get("lineno"))
                end_line = self._as_int(data.get("end_lineno"))
                start_col = self._as_int(data.get("col_offset"))
                end_col = self._as_int(data.get("end_col_offset"))

                span = TextSpan(
                    start_line=start_line + offset if start_line is not None else -1,
                    start_col=start_col if start_col is not None else -1,
                    end_line=end_line + offset if end_line is not None else -1,
                    end_col=end_col if end_col is not None else -1,
                )

                if span.is_valid():
                    nodes[data["source"]] = span
                    nodes[data["target"]] = span
        return nodes

    def _parse_golden_strings(self, json_data: GoldenGraph) -> dict[str, str]:
        """Creates a lookup of Golden Node ID -> Raw Code String for fallback matching."""
        raw_map = {}
        for element in json_data.get("elements", []):
            data: GoldenElementData = element.get("data", {})
            if "id" in data and "code" in data and data["code"]:
                raw_map[data["id"]] = str(data["code"]).strip()
        return raw_map

    def find_best_match(self, vamsa_id: str, vamsa_span: TextSpan, threshold: float = 0.5) -> str | None:
        """Finds the Golden Node ID using the Overlap Coefficient."""
        best_match_id = None

        if vamsa_span.is_valid():
            max_overlap = 0.0
            for golden_id, golden_span in self.golden_nodes_spatial.items():
                overlap = vamsa_span.overlap_coefficient(golden_span)
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_match_id = golden_id

            if max_overlap >= threshold:
                return best_match_id

        vamsa_clean_id = vamsa_id.split(":")[0]
        for golden_id, golden_code in self.golden_nodes_raw.items():
            if vamsa_clean_id == golden_code:
                return golden_id

        return None
