import ast

import networkx as nx

from .core import reset_vamsa_counter
from .extraction import GenWIR
from .lineage import KB, AnnotationWIR, track_provenance


def execute_vamsa_pipeline(
    ast_tree: ast.AST,
    what_track: set[str] | None = None,
    kb_csv_path: str | None = None,
) -> tuple[nx.DiGraph, set[str], set[str]]:
    if what_track is None:
        what_track = {"features"}
    reset_vamsa_counter()

    wir_graph, prs, _tuples = GenWIR(ast_tree)

    kb = KB(kb_csv_path=kb_csv_path)
    annotator = AnnotationWIR(wir_graph, prs, kb)
    annotated_wir = annotator.annotate()

    c_plus, c_minus = track_provenance(annotator, prs[::-1], what_track=what_track)

    return annotated_wir, c_plus, c_minus


__all__ = ["execute_vamsa_pipeline"]
