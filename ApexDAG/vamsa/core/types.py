import ast
from dataclasses import dataclass
from typing import Union

WIRNodeType = Union[str, list[str], ast.AST, list[ast.AST]]
PRType = tuple[str, str, str, str]

@dataclass
class WIRNode:
    """
    Data container for WIR elements during AST traversal.
    """
    node: WIRNodeType
    isAttribute: bool = False
