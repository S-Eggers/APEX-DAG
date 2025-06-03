import ast
import os

from ApexDAG.vamsa.generate_wir import GenWIR
from ApexDAG.vamsa.annotate_wir import AnnotationWIR, KB

# todo: move to utils
def get_name(node):
    return node.split(':')[0]

def is_constant(var, prs):
    """
    Check if a variable is a constant based on its annotations.
    Returns True if the variable is a constant, False otherwise.
    Var is constant if it is not an output of any other PR!
    """
    if var is None:
        return False
    
    for pr in prs:
        _, _, _, output_nodes = pr
        if output_nodes is None:
            continue
        if isinstance(output_nodes, list):
            if var in output_nodes:
                return False
        else:
            if var == output_nodes:
                return False
    return True

def drop_traversal(pr, tracker):
    """
    Traversal rule for DataFrame.drop operation:
    Follow the input variable representing the columns to drop (if any).
    Assumes pr.input_vars[0] is the DataFrame and remaining inputs are columns/rows.
    """
    input_nodes, _, _, _ = pr
    next_prs = []
    input_nodes = input_nodes if isinstance(input_nodes, list) else [input_nodes]
    for var in input_nodes:
        if var in tracker.var_to_pr:
            for nextpr in tracker.var_to_pr[var]:
                next_prs.append(nextpr)
    return next_prs

def list_traversal(pr, tracker):

    input_nodes, _, _, _ = pr
    next_prs = []
    input_nodes = input_nodes if isinstance(input_nodes, list) else [input_nodes]
    for var in input_nodes:
        if var in tracker.var_to_pr:
            for nextpr in tracker.var_to_pr[var]:
                next_prs.append(nextpr)
    return next_prs

def iloc_traversal(pr, tracker):
    """
    Traversal rule for DataFrame.iloc operation:
    Follow the input variable representing the indexing expression (if any).
    Assumes pr.input_vars[0] is the DataFrame and pr.input_vars[1] is the index/slice var.
    """
    _, _, _, output_nodes = pr
    output_nodes = output_nodes if isinstance(output_nodes, list) else [output_nodes]
    next_prs = []
    for output_node in output_nodes:
        idx_var = output_node
        if idx_var in tracker.cal_to_pr:
            for next_pr in tracker.cal_to_pr[idx_var]:
                next_prs.append(next_pr)
    return next_prs

def subscript_traversal(pr, tracker):
    """
    Traversal rule for Subscript operation (e.g., df[var] or df[var_slice]):
    Follow the input variable representing the key/index (if any).
    Assumes pr.input_vars[0] is the DataFrame and pr.input_vars[1] is the key/slice var.
    """
    input_nodes, _, _, _ = pr
    input_nodes = input_nodes if isinstance(input_nodes, list) else [input_nodes]
    next_prs = []
    for input_node in input_nodes:
        if input_node in tracker.var_to_pr:
            for next_pr in tracker.var_to_pr[input_node]:
                next_prs.append(next_pr)
    return next_prs

def slice_traversal(pr, tracker):
    """
    Traversal rule for a Slice node:
    Follow input edges for lower/upper bound variables (if any).
    Assumes pr.input_vars[0] is the sequence being sliced,
    pr.input_vars[1] is the lower bound var (optional),
    pr.input_vars[2] is the upper bound var (optional).
    """
    input_nodes, _, _, _ = pr
    input_nodes = input_nodes if isinstance(input_nodes, list) else [input_nodes]
    next_prs = []
    for bound_var in input_nodes[1:]:
        if bound_var in tracker.var_to_pr:
            next_prs.append(tracker.var_to_pr[bound_var])
    return next_prs

# Knowledge base: map operation name to column_exclusion flag and traversal rule
KBC = {
    'drop': {'column_exclusion': True, 'traversal_rule': drop_traversal},
    'iloc': {'column_exclusion': False, 'traversal_rule': iloc_traversal},
    'Subscript': {'column_exclusion': False, 'traversal_rule': subscript_traversal},
    'Slice': {'column_exclusion': False, 'traversal_rule': slice_traversal},
    'List': {'column_exclusion': False, 'traversal_rule': list_traversal},
}


class ProvenanceTracker:
    """
    Provenance Tracker for identifying included/excluded columns.
    Accepts an annotated WIR (with variable annotations) and a list of PRs.
    It uses the KBC to guide traversal and returns two sets:
      - C_plus  : columns included in features/labels
      - C_minus : columns excluded from features/labels
    """
    def __init__(self, wir, prs, kbc=KBC):

        self.wir = wir
        self.prs = prs
        self.kbc = kbc
        
        self.var_to_pr = {}
        self.cal_to_pr = {}
        for pr in prs:
            _, caller, _, output_nodes = pr
            if caller in self.cal_to_pr:
                self.cal_to_pr[caller].append(pr)
            else:
                self.cal_to_pr[caller] = [pr]
            if output_nodes is not None:
                if isinstance(output_nodes, list):
                    for output_node in output_nodes:
                        if output_node in self.var_to_pr:
                            self.var_to_pr[output_node].append(pr)
                        else:
                            self.var_to_pr[output_node] = [pr]
                            
                if output_nodes in self.var_to_pr:
                    self.var_to_pr[output_nodes].append(pr)
                else:
                    self.var_to_pr[output_nodes] = [pr]
        self.C_plus = set()
        self.C_minus = set()

        self.visited_prs = set()

    def track(self, what_track):
        """
        Execute the provenance tracking algorithm (PTracker, see Fig.6 in VAMSA).
        Iterates over PRs, finds those related to features/labels, and applies GuideEval.
        Returns (C_plus, C_minus).
        """
        self.C_plus.clear()
        self.C_minus.clear()
        self.visited_prs.clear()

        annotations = self.wir.annotated_wir.nodes
        feature_labels = what_track
        for pr in self.prs:
            input_nodes, _, operation_node, output_nodes = pr
            vars_to_check = input_nodes if isinstance(input_nodes, list) else [input_nodes]
            if output_nodes:
                if isinstance(output_nodes, list):
                    vars_to_check.extend(output_nodes)
                else:
                    vars_to_check.append(output_nodes)
            annotated = False
            for var in vars_to_check:
                info = annotations.get(var)
                if info is not None:
                    ann = annotations[var].get('annotations')
                    ann = ann[0] if isinstance(ann, list) else ann
                    if (ann is not None) and (ann in feature_labels):
                            annotated = True
            if not annotated:
                continue

            # get name
            print(f"Processing PR: {pr}")
            operation_name = get_name(operation_node)
            if operation_name in self.kbc:
                self._guide_eval(pr)
                
        return self.C_plus, self.C_minus

    def _guide_eval(self, pr, col_excl=None):
        """
        Recursive GuideEval operator (Figure 6 in VAMSA).
        It updates C_plus/C_minus based on constants or further traversals.
        """
        input_nodes, _, operation_node, _ = pr
        input_nodes = input_nodes if isinstance(input_nodes, list) else [input_nodes]
        if pr in self.visited_prs:
            return
        self.visited_prs.add(pr)
        op_name = get_name(operation_node)
        entry = self.kbc.get(op_name)
        if not entry:
            return
        if col_excl is None:
            col_excl = entry['column_exclusion']
        traversal_rule = entry['traversal_rule']

        constant_inputs = [var for var in input_nodes if is_constant(var, self.prs)]
        for cnst in constant_inputs:
                if isinstance(cnst, (list, tuple)):
                    for col in cnst:
                        if col_excl:
                            self.C_minus.add(col)
                        else:
                            self.C_plus.add(col)
                else:
                    if isinstance(cnst, slice):
                        col = (cnst.start, cnst.stop)
                    else:
                        col = cnst
                    if col_excl:
                        self.C_minus.add(col)
                    else:
                        self.C_plus.add(col)
        if len(constant_inputs) == len(input_nodes):
            return

        next_prs = traversal_rule(pr, self)
        for next_pr in next_prs:
            self._guide_eval(next_pr, col_excl=col_excl)

def track_provenance(annotated_wir, prs, what_track):
    tracker = ProvenanceTracker(annotated_wir, prs)
    C_plus, C_minus = tracker.track(what_track=what_track)
    return C_plus, C_minus


if __name__ == "__main__":
    file_path = 'data/raw/test_vamsa.py'
    location_related_attributes = ['lineno', 'col_offset', 'end_lineno', 'end_col_offset']
    
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(file_path, 'r') as file:
        file_content = file.read()
    file_lines = file_content.split('\n')
    
    parsed_ast = ast.parse(file_content)
    wir, prs, tuples = GenWIR(parsed_ast, output_filename=output_dir+'/wir-unannotated-test.png', if_draw_graph=True)

    annotated_wir = AnnotationWIR(wir, prs, KB(None))
    annotated_wir.annotate()

    input_nodes, caller_nodes, operation_nodes, output_nodes= tuples

    what_track = {'features'}  # 'labels' # Specify what to track
    C_plus, C_minus = track_provenance(
        annotated_wir,
        prs[::-1],
        what_track=what_track
    )
    print(f"Columns included in {what_track} (C_plus):", C_plus)
    print(f"Columns excluded from {what_track} (C_minus):", C_minus)
