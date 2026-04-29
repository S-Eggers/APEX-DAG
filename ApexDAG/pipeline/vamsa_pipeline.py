import ast
from typing import Any

import networkx as nx

from ApexDAG.sca.constants import DOMAIN_NODE_TYPES
from ApexDAG.serializer.vamsa_serializer import VamsaSerializer
from ApexDAG.vamsa import execute_vamsa_pipeline


class VamsaPipeline:
    def __init__(
        self, serializer: VamsaSerializer, kb_csv_path: str | None = None
    ) -> None:
        self.serializer = serializer
        self.kb_csv_path = kb_csv_path

    def _apply_domain_labels(self, annotated_wir: nx.DiGraph) -> None:
        """
        Translates raw Vamsa KB string annotations into the strict integer taxonomy.
        """
        for node, data in annotated_wir.nodes(data=True):
            semantic_type = DOMAIN_NODE_TYPES["NOT_RELEVANT"]

            annotations = data.get("annotations", [])
            ann_strings = [str(a).lower() for a in annotations]

            if "model" in ann_strings:
                semantic_type = DOMAIN_NODE_TYPES["MODEL"]
            elif any(
                k in ann_strings
                for k in ["data", "dataset", "features", "target", "dataframe"]
            ):
                semantic_type = DOMAIN_NODE_TYPES["DATASET"]

            annotated_wir.nodes[node]["node_type"] = semantic_type

    def execute(
        self, cells: list, what_track: set[str] | None = None
    ) -> dict[str, Any]:
        """
        Executes the Vamsa lineage pipeline and serializes the result for the frontend.
        """
        if what_track is None:
            what_track = {"features"}

        code_string = "\n".join([cell.get("source", "") for cell in cells])

        try:
            ast_tree = ast.parse(code_string)
        except SyntaxError as e:
            raise ValueError(f"Vamsa Pipeline failed to parse code: {e}") from e

        annotated_wir, c_plus, c_minus = execute_vamsa_pipeline(
            ast_tree, what_track=what_track, kb_csv_path=self.kb_csv_path
        )
        self._apply_domain_labels(annotated_wir)

        payload = self.serializer.to_dict(annotated_wir)
        payload["metadata"] = {
            "vamsa_provenance": {"c_plus": list(c_plus), "c_minus": list(c_minus)}
        }

        return payload
