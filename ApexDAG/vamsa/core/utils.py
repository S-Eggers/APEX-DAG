import logging
from typing import Set, List
from .types import PRType

logger = logging.getLogger(__name__)

_vamsa_id_counter = 0

def reset_vamsa_counter() -> None:
    global _vamsa_id_counter
    _vamsa_id_counter = 0

def add_id() -> str:
    global _vamsa_id_counter
    _vamsa_id_counter += 1
    return f":id{_vamsa_id_counter}"

def remove_id(node_name: str | None) -> str:
    if node_name is None:
        return ""
    return node_name.split(":")[0]

def is_empty_or_none_list(I) -> bool:
    return I is None or (isinstance(I, list) and all(x is None for x in I))

def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def check_bipartie(PRs: Set[PRType]) -> bool:
    """
    Validates if the generated Process Relations form a bipartite graph.
    """
    operations = set([pr[2] for pr in PRs])
    other = set(flatten([[pr[0], pr[1], pr[3]] for pr in PRs]))
    intersections = operations.intersection(other)
    
    if len(intersections) > 0:
        logger.warning(f"Bipartite violation detected. Intersections: {intersections}")
        
    return len(intersections) == 0

def get_relevant_code(node, file_lines) -> str | None:
    """
    Extracts the raw code segment for a given AST node.
    """
    if hasattr(node, "__dict__"):
        if "lineno" in vars(node) and "col_offset" in vars(node) and "end_col_offset" in vars(node):
            return file_lines[node.lineno - 1][node.col_offset : node.end_col_offset]
    return None

def merge_prs(P: List[PRType], P_prime: List[PRType]) -> Set[PRType]:
    """
    Deduplicates and merges two sets of Process Relations while preserving order.
    """
    return list(dict.fromkeys(P + P_prime).keys())