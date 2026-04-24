import ast
from dataclasses import dataclass
from typing import List, Union, Tuple

WIRNodeType = Union[str, List[str], ast.AST, List[ast.AST]]
PRType = Tuple[str, str, str, str]

@dataclass
class WIRNode:
    """
    Data container for WIR elements during AST traversal.
    """
    node: WIRNodeType
    isAttribute: bool = False