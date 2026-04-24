from .core import reset_vamsa_counter
from .extraction import GenWIR
from .lineage import KB, AnnotationWIR, track_provenance


def execute_vamsa_pipeline(ast_tree, what_track={"features"}, kb_csv_path=None):
    reset_vamsa_counter()
    
    G, prs, tuples = GenWIR(ast_tree)
    
    kb = KB(kb_csv_path=kb_csv_path)
    annotated_wir = AnnotationWIR(G, prs, kb)
    annotated_wir.annotate()
    
    c_plus, c_minus = track_provenance(annotated_wir, prs[::-1], what_track=what_track)
    
    return G, c_plus, c_minus

__all__ = ["execute_vamsa_pipeline"]