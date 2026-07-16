import logging
from collections.abc import Generator
from typing import Any

from .types import PRType

logger = logging.getLogger(__name__)

def reset_vamsa_counter() -> None:
    global _vamsa_id_counter
    _vamsa_id_counter = 0

_vamsa_id_counter = 0

def add_id() -> str:
    global _vamsa_id_counter
    _vamsa_id_counter += 1
    return f":id{_vamsa_id_counter}"

def remove_id(node_name: str | None) -> str:
    if node_name is None:
        return ""
    return node_name.split(":")[0]

def is_empty_or_none_list(value: object) -> bool:
    return value is None or (isinstance(value, list) and all(x is None for x in value))

def flatten(lst: list) -> Generator[Any, None, None]:
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item

def check_bipartie(prs: set[PRType]) -> bool:
    """Validates if the generated Process Relations form a bipartite graph."""
    operations = {pr[2] for pr in prs}
    other = set(flatten([[pr[0], pr[1], pr[3]] for pr in prs]))
    intersections = operations.intersection(other)

    if len(intersections) > 0:
        logger.warning("Bipartite violation detected. Intersections: %s", intersections)

    return len(intersections) == 0

def get_relevant_code(node: object, file_lines: list[str]) -> str | None:
    """Extracts the raw code segment for a given AST node."""
    if hasattr(node, "__dict__") and ("lineno" in vars(node) and "col_offset" in vars(node) and "end_col_offset" in vars(node)):
        return file_lines[node.lineno - 1][node.col_offset : node.end_col_offset]
    return None

def merge_prs(p: list[PRType], p_prime: list[PRType]) -> set[PRType]:
    """Deduplicates and merges two sets of Process Relations while preserving order."""
    return list(dict.fromkeys(p + p_prime).keys())
