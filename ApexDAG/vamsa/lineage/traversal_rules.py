
from ..core.types import PRType


def is_constant(var: str, prs: list[PRType]) -> bool:
    """
    Check if a variable is a constant. Var is constant if it is not an output of any other PR.
    """
    if var is None:
        return False
    for pr in prs:
        _, _, _, output_nodes = pr
        if output_nodes is None: continue
        if (isinstance(output_nodes, list) and var in output_nodes) or var == output_nodes: return False
    return True

def drop_traversal(pr, tracker):
    input_nodes, _, _, _ = pr
    next_prs = []
    input_nodes = input_nodes if isinstance(input_nodes, list) else [input_nodes]
    for var in input_nodes:
        if var in tracker.var_to_pr:
            for nextpr in tracker.var_to_pr[var]: next_prs.append(nextpr)
    return next_prs

def list_traversal(pr, tracker):
    return drop_traversal(pr, tracker)

def keyword_traversal(pr, tracker):
    input_nodes, _, _, output_node = pr
    if "label" not in output_node: return []
    return drop_traversal(pr, tracker)

def iloc_traversal(pr, tracker):
    _, _, _, output_nodes = pr
    output_nodes = output_nodes if isinstance(output_nodes, list) else [output_nodes]
    next_prs = []
    for output_node in output_nodes:
        if output_node in tracker.cal_to_pr:
            for next_pr in tracker.cal_to_pr[output_node]: next_prs.append(next_pr)
    return next_prs

def subscript_traversal(pr, tracker):
    return drop_traversal(pr, tracker)

def slice_traversal(pr, tracker):
    input_nodes, _, _, _ = pr
    input_nodes = input_nodes if isinstance(input_nodes, list) else [input_nodes]
    next_prs = []
    for bound_var in input_nodes[1:]:
        if bound_var in tracker.var_to_pr:
            next_prs.append(tracker.var_to_pr[bound_var])
    return next_prs

# Vamsa's Static Knowledge Base Rules
KBC = {
    "drop": {"column_exclusion": True, "traversal_rule": drop_traversal},
    "iloc": {"column_exclusion": False, "traversal_rule": iloc_traversal},
    "Subscript": {"column_exclusion": False, "traversal_rule": subscript_traversal},
    "Slice": {"column_exclusion": False, "traversal_rule": slice_traversal},
    "List": {"column_exclusion": False, "traversal_rule": list_traversal},
    "keyword": {"column_exclusion": False, "traversal_rule": keyword_traversal},
}
