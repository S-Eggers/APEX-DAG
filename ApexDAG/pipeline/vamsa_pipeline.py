import ast
from typing import Any

from ApexDAG.serializer.vamsa_serializer import VamsaSerializer
from ApexDAG.vamsa import execute_vamsa_pipeline


class VamsaPipeline:
    def __init__(self, serializer: VamsaSerializer, kb_csv_path: str = None):
        self.serializer = serializer
        self.kb_csv_path = kb_csv_path

    def execute(self, cells: list, what_track: set[str] = None) -> dict[str, Any]:
        """
        Executes the Vamsa lineage pipeline and serializes the result for the frontend.
        """
        if what_track is None:
            what_track = {"features"}

        code_string = "\n".join([cell.get("source", "") for cell in cells])

        try:
            ast_tree = ast.parse(code_string)
        except SyntaxError as e:
            raise ValueError(f"Vamsa Pipeline failed to parse code: {e}")

        G, c_plus, c_minus = execute_vamsa_pipeline(
            ast_tree,
            what_track=what_track,
            kb_csv_path=self.kb_csv_path
        )

        payload = self.serializer.to_dict(G)
        payload["metadata"] = {
            "vamsa_provenance": {
                "c_plus": list(c_plus),
                "c_minus": list(c_minus)
            }
        }

        return payload
