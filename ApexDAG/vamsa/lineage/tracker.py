import logging
from typing import Tuple, Set, List

from ..core.utils import remove_id
from ..core.types import PRType
from .traversal_rules import KBC, is_constant
from .annotator import AnnotationWIR
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)

class ProvenanceTracker:
    """
    Executes the PTracker algorithm to identify included/excluded columns based on the WIR.
    """
    def __init__(self, wir: AnnotationWIR, prs: List[PRType], kbc=KBC):
        self.wir = wir
        self.prs = prs
        self.kbc = kbc

        self.var_to_pr = {}
        self.cal_to_pr = {}
        
        for pr in prs:
            _, caller, _, output_nodes = pr
            if caller in self.cal_to_pr: self.cal_to_pr[caller].append(pr)
            else: self.cal_to_pr[caller] = [pr]
            
            if output_nodes is not None:
                out_list = output_nodes if isinstance(output_nodes, list) else [output_nodes]
                for out_node in out_list:
                    if out_node in self.var_to_pr: self.var_to_pr[out_node].append(pr)
                    else: self.var_to_pr[out_node] = [pr]
                    
        self.C_plus = set()
        self.C_minus = set()
        self.visited_prs = set()

    def track(self, what_track: set) -> Tuple[Set, Set]:
        self.C_plus.clear()
        self.C_minus.clear()
        self.visited_prs.clear()

        annotations = self.wir.annotated_wir.nodes
        feature_labels = what_track
        
        for pr in self.prs:
            input_nodes, _, operation_node, output_nodes = pr
            vars_to_check = input_nodes if isinstance(input_nodes, list) else [input_nodes]
            if output_nodes:
                out_list = output_nodes if isinstance(output_nodes, list) else [output_nodes]
                vars_to_check.extend(out_list)
                
            annotated = False
            for var in vars_to_check:
                info = annotations.get(var)
                if info is not None:
                    ann = info.get("annotations")
                    if ann:
                        ann_val = ann[0] if isinstance(ann, list) else ann
                        if ann_val in feature_labels:
                            annotated = True
                            
            if not annotated: 
                continue

            operation_name = remove_id(operation_node)
            if operation_name in self.kbc:
                self._guide_eval(pr)

        return self.C_plus, self.C_minus

    def _guide_eval(self, pr: PRType, col_excl=None):
        input_nodes, _, operation_node, out_nodes = pr
        input_nodes = input_nodes if isinstance(input_nodes, list) else [input_nodes]
        
        if pr in self.visited_prs and col_excl is None: 
            return
            
        self.visited_prs.add(pr)
        
        op_name = remove_id(operation_node)
        entry = self.kbc.get(op_name)
        if not entry: return
        
        if col_excl is None: col_excl = entry["column_exclusion"]
        traversal_rule = entry["traversal_rule"]

        constant_inputs = [var for var in input_nodes if is_constant(var, self.prs)]
        if ("keyword" in op_name) and "label" not in out_nodes:
            constant_inputs = []
            
        for cnst in constant_inputs:
            cols = cnst if isinstance(cnst, (list, tuple)) else [(cnst.start, cnst.stop) if isinstance(cnst, slice) else cnst]
            for col in cols:
                if col_excl:
                    self.C_minus.add(col)
                    if col in self.C_plus: self.C_plus.remove(col)
                else:
                    self.C_plus.add(col)

        if len(constant_inputs) == len(input_nodes): 
            return

        next_prs = traversal_rule(pr, self)
        for next_pr in next_prs:
            self._guide_eval(next_pr, col_excl=col_excl)


def track_provenance(annotated_wir: AnnotationWIR, prs: List[PRType], what_track: set) -> Tuple[Set, Set]:
    """
    Public entry point for lineage tracking.
    """
    tracker = ProvenanceTracker(annotated_wir, prs)
    return tracker.track(what_track=what_track)