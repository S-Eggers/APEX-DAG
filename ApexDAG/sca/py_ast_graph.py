import ast

from ApexDAG.notebook import Notebook
from ApexDAG.sca.ast_graph import ASTGraph
from ApexDAG.sca.constants import AST_EDGE_TYPES, AST_NODE_TYPES
from ApexDAG.sca.models import GraphEdge, GraphNode


class PythonASTGraph(ASTGraph, ast.NodeVisitor):
    def create_notebook_root(self) -> int:
        """
        Implements the master root node to prevent a disconnected AST forest.
        """
        node_label = "Notebook"
        numeric_type = AST_NODE_TYPES.get("Module", 100)
        cell_context = getattr(self, "current_cell_id", "global_notebook")

        node_model = GraphNode(
            id=self.node_counter,
            label=node_label,
            node_type=numeric_type,
            cell_id=cell_context,
            code="<notebook_root>",
        )

        self._G.add_node(node_model.id, **node_model.to_networkx_attrs())

        node_id = self.node_counter
        self.node_counter += 1
        return node_id

    def connect_notebook_root(self, root_id: int, cell_module_id: int) -> None:
        """
        Connects the master root to the individual cell modules.
        """
        self.add_edge(
            source=root_id,
            target=cell_module_id,
            edge_type=AST_EDGE_TYPES.get("AST_PARENT_CHILD", 0),
            label="cell_module",
        )

    def generic_visit(self, node: ast.AST) -> int:
        """
        Visits a given AST node, processes it, and builds a directed graph representation.
        """
        node_id: int = self.add_node(node)

        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST) and not isinstance(item, (ast.Load, ast.Store)):
                        child_id = self.visit(item)
                        self.add_edge(
                            source=node_id,
                            target=child_id,
                            edge_type=AST_EDGE_TYPES["AST_PARENT_CHILD"],
                            label=field,
                            lineno=getattr(item, "lineno", None),
                            col_offset=getattr(item, "col_offset", None),
                            end_lineno=getattr(item, "end_lineno", None),
                            end_col_offset=getattr(item, "end_col_offset", None),
                        )
            elif isinstance(value, ast.AST) and not isinstance(value, (ast.Load, ast.Store)):
                child_id = self.visit(value)
                self.add_edge(
                    source=node_id,
                    target=child_id,
                    edge_type=AST_EDGE_TYPES["AST_PARENT_CHILD"],
                    label=field,
                    lineno=getattr(value, "lineno", None),
                    col_offset=getattr(value, "col_offset", None),
                    end_lineno=getattr(value, "end_lineno", None),
                    end_col_offset=getattr(value, "end_col_offset", None),
                )

        return node_id

    def add_node(self, node: ast.AST) -> int:
        node_label = type(node).__name__
        code = self.get_code_from_node(node) if hasattr(node, "lineno") else ""
        numeric_type = AST_NODE_TYPES.get(node_label, AST_NODE_TYPES["AST_UNKNOWN"])
        cell_context = getattr(self, "current_cell_id", "unknown_cell")

        node_model = GraphNode(
            id=self.node_counter,
            label=node_label,
            node_type=numeric_type,
            cell_id=cell_context,
            code=code,
            lineno=getattr(node, "lineno", None),
            col_offset=getattr(node, "col_offset", None),
            end_lineno=getattr(node, "end_lineno", None),
            end_col_offset=getattr(node, "end_col_offset", None),
        )

        self._G.add_node(node_model.id, **node_model.to_networkx_attrs())

        node_id = self.node_counter
        self.node_counter += 1
        return node_id

    def add_edge(
        self,
        source: int,
        target: int,
        edge_type: int,
        label: str = "edge",
        lineno: int | None = None,
        col_offset: int | None = None,
        end_lineno: int | None = None,
        end_col_offset: int | None = None,
    ) -> None:
        cell_context = getattr(self, "current_cell_id", "unknown_cell")

        edge_model = GraphEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            cell_id=cell_context,
            label=label,
            lineno=lineno,
            col_offset=col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
        )

        self._G.add_edge(edge_model.source, edge_model.target, **edge_model.to_networkx_attrs())

    @staticmethod
    def from_notebook_windows(notebook: Notebook) -> list["PythonASTGraph"]:
        """
        Creates a list of ASTGraph objects from code cells of a given notebook.
        """
        ast_graphs = []
        for cell_window in notebook:
            G = PythonASTGraph()
            G.parse_code(notebook.cell_code(cell_window))
            ast_graphs.append(G)

        return ast_graphs
