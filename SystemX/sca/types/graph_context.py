import ast
from typing import Protocol

from SystemX.state import Stack, State

class GraphContext(Protocol):
    """Strict contract for graph mutation."""

    @property
    def current_state(self) -> State: ...

    @current_state.setter
    def current_state(self, value: State) -> None: ...

    @property
    def state_stack(self) -> Stack: ...

    @property
    def current_cell_source(self) -> str: ...

    @property
    def current_cell_id(self) -> str: ...

    def visit(self, node: ast.AST) -> ast.AST: ...

    def add_node(self, node_name: str, node_type: int, code: str = "") -> None: ...

    def add_edge(
        self,
        source: str,
        target: str,
        label: str,
        edge_type: int,
        raw_code: str = "",
        lineno: int = -1,
        col_offset: int = -1,
        end_lineno: int = -1,
        end_col_offset: int = -1,
    ) -> None: ...
