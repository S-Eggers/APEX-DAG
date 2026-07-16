import ast
from dataclasses import dataclass

WIRNodeType = str | list[str] | ast.AST | list[ast.AST]
PRType = tuple[str, str, str, str]


@dataclass
class SpatialMetadata:
    lineno: int | None = None
    col_offset: int | None = None
    end_lineno: int | None = None
    end_col_offset: int | None = None


class WIRNode:
    def __init__(self, node: ast.AST, is_attribute: bool = False, parent_ast: ast.AST | None = None) -> None:
        self.node = node
        self.is_attribute = is_attribute
        self.spatial = SpatialMetadata()

        if isinstance(node, ast.AST):
            self._extract_spatial(node)
        elif parent_ast is not None and isinstance(parent_ast, ast.AST):
            self._extract_spatial(parent_ast)

    def _extract_spatial(self, target: ast.AST) -> None:
        self.spatial.lineno = getattr(target, "lineno", None)
        self.spatial.col_offset = getattr(target, "col_offset", None)
        self.spatial.end_lineno = getattr(target, "end_lineno", None)
        self.spatial.end_col_offset = getattr(target, "end_col_offset", None)

    def __hash__(self) -> int:
        if isinstance(self.node, list):
            return hash(tuple(self.node))
        return hash(self.node)

    def __eq__(self, other: "WIRNode") -> bool:
        if not isinstance(other, WIRNode):
            return False
        return self.node == other.node
