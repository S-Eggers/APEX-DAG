import ast
from typing import Dict, Any

class ComplexityVisitor(ast.NodeVisitor):
    """
    AST Visitor to extract advanced code complexity metrics.
    Tracks structural frequencies and maximum control-flow nesting depth.
    """
    def __init__(self) -> None:
        self.metrics: Dict[str, int] = {
            "loops": 0,
            "for_else": 0,
            "while_else": 0,
            "branches": 0,
            "match_cases": 0,
            "list_comp": 0,
            "dict_comp": 0,
            "set_comp": 0,
            "gen_expr": 0,
            "try_except": 0,
            "with_blocks": 0,
            "max_nesting_depth": 0
        }
        self._current_depth: int = 0

    def _enter_nested_block(self) -> None:
        """Increments current depth and updates the maximum observed depth."""
        self._current_depth += 1
        if self._current_depth > self.metrics["max_nesting_depth"]:
            self.metrics["max_nesting_depth"] = self._current_depth

    def _exit_nested_block(self) -> None:
        """Decrements current depth when exiting a structural block."""
        self._current_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        self.metrics["loops"] += 1
        if node.orelse:
            self.metrics["for_else"] += 1
            
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.metrics["loops"] += 1
        if node.orelse:
            self.metrics["for_else"] += 1
            
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_While(self, node: ast.While) -> None:
        self.metrics["loops"] += 1
        if node.orelse:
            self.metrics["while_else"] += 1
            
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_If(self, node: ast.If) -> None:
        self.metrics["branches"] += 1
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_Match(self, node: ast.Match) -> None:
        self.metrics["match_cases"] += len(node.cases)
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.metrics["list_comp"] += 1
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.metrics["dict_comp"] += 1
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.metrics["set_comp"] += 1
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.metrics["gen_expr"] += 1
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_Try(self, node: ast.Try) -> None:
        self.metrics["try_except"] += 1
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def visit_With(self, node: ast.With) -> None:
        self.metrics["with_blocks"] += 1
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()
        
    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self.metrics["with_blocks"] += 1
        self._enter_nested_block()
        self.generic_visit(node)
        self._exit_nested_block()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.metrics!r})"