from .core import reset_vamsa_counter
from .extraction import GenWIR
from .lineage import KB, AnnotationWIR, track_provenance


def execute_vamsa_pipeline(ast_tree, what_track=None, kb_csv_path=None):
    if what_track is None:
        what_track = {"features"}
    reset_vamsa_counter()

    G, prs, _tuples = GenWIR(ast_tree)

    kb = KB(kb_csv_path=kb_csv_path)
    annotated_wir = AnnotationWIR(G, prs, kb)
    annotated_wir.annotate()

    c_plus, c_minus = track_provenance(annotated_wir, prs[::-1], what_track=what_track)

    return G, c_plus, c_minus


__all__ = ["execute_vamsa_pipeline"]
