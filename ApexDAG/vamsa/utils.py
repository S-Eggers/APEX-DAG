import ast
import logging
import random
from dataclasses import dataclass

from typing import List, Union, Set, Tuple

logging.basicConfig(level=logging.WARNING)  # Adjust for debugging
logger = logging.getLogger(__name__)  # Get a logger for this module

WIRNodeType = Union[str, List[str], ast.AST, List[ast.AST]]
PRType = Tuple[str, str, str, str]

@dataclass
class WIRNode:
    node: WIRNodeType
    isAttribute: bool = False

def check_bipartie(PRs: Set[PRType]) -> bool:
    '''
    Checks if the PRs are bipartite
    '''
    operations = set([pr[2] for pr in PRs])
    other = set(flatten([[pr[0],pr[1],pr[3]] for pr in PRs]))
    intersections = operations.intersection(other)
    # log 
    if len(intersections) > 0:
        logger.warning(f"Intersections: {intersections}")
    return len(intersections) == 0

def add_id():
    return ':id' + str(random.randint(0, 2500))

def remove_id(node_name: str | None):
    if node_name is None:
        return ''
    return node_name.split(':')[0]

def is_empty_or_none_list(I):
    return I is None or (isinstance(I, list) and all(x is None for x in I))

def flatten(lst):
    """
    Recursively flattens a list while preserving strings.
    
    :param lst: List to be flattened.
    :return: Flattened list.
    """
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item
            

def get_relevant_code(node, file_lines):
    '''
    Utility, returns code pertaining to a node
    '''
    if hasattr(node, '__dict__'):
        if 'lineno' in vars(node) and 'col_offset' in vars(node) and 'end_col_offset' in vars(node):
            return file_lines[node.lineno-1][node.col_offset:node.end_col_offset]

# delete, only for inspection
def print_relevant_code(element):
    '''
    Utility, prints code pertaining to a node
    '''
    if isinstance(element, list):
        for el in element:
            print_relevant_code(el)
    elif isinstance(element, str):
        print(element)
    elif element is not None:
        get_relevant_code(element)
    else:
        print("None")           
            
def remove_comment_lines(code_string: str):
    '''
    Removes comments and pure print statements...
    '''
    code_lines = [line for line in code_string.splitlines() if not line.strip().startswith('#')]
    code_lines = [line for line in code_lines if not line.startswith('print')]
    return '\n'.join(code_lines)

def merge_prs(P: List[PRType], P_prime: List[PRType]) -> Set[PRType]:
    return list(dict.fromkeys(P + P_prime).keys())