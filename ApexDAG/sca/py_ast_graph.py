import ast
from typing import List
from ApexDAG.notebook import Notebook

from ApexDAG.sca.ast_graph import ASTGraph
from ApexDAG.util.draw import Draw
from ApexDAG.sca.constants import AST_NODE_TYPES, AST_EDGE_TYPES
from ApexDAG.sca.models import GraphNode, GraphEdge


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
            code="<notebook_root>"
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
            label="cell_module"
        )

    def generic_visit(self, node: ast.AST) -> int:
        """
        Visits a given AST node, processes it, and builds a directed graph representation.
        """
        node_id: int = self.add_node(node)
        
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST) and not isinstance(
                        item, (ast.Load, ast.Store)
                    ):
                        child_id = self.visit(item)
                        self.add_edge(
                            source=node_id, 
                            target=child_id, 
                            edge_type=AST_EDGE_TYPES["AST_PARENT_CHILD"], 
                            label=field
                        )
            elif isinstance(value, ast.AST) and not isinstance(
                value, (ast.Load, ast.Store)
            ):
                child_id = self.visit(value)
                self.add_edge(
                    source=node_id, 
                    target=child_id, 
                    edge_type=AST_EDGE_TYPES["AST_PARENT_CHILD"], 
                    label=field
                )

        return node_id

    def add_node(self, node) -> int:
        node_label = type(node).__name__
        code = self.get_code_from_node(node) if hasattr(node, "lineno") else ""
        numeric_type = AST_NODE_TYPES.get(node_label, AST_NODE_TYPES["AST_UNKNOWN"])
        cell_context = getattr(self, "current_cell_id", "unknown_cell")

        node_model = GraphNode(
            id=self.node_counter,
            label=node_label,
            node_type=numeric_type,
            cell_id=cell_context,
            code=code
        )

        self._G.add_node(node_model.id, **node_model.to_networkx_attrs())
        
        node_id = self.node_counter
        self.node_counter += 1
        return node_id

    def add_edge(self, source: int, target: int, edge_type: int, label: str = "edge") -> None:
        cell_context = getattr(self, "current_cell_id", "unknown_cell")

        edge_model = GraphEdge(
            source=source,
            target=target,
            edge_type=edge_type,
            cell_id=cell_context,
            label=label
        )

        self._G.add_edge(
            edge_model.source, 
            edge_model.target, 
            **edge_model.to_networkx_attrs()
        )

    def to_json(self):
        draw = Draw(None, None)
        G = self.get_graph()
        return draw.ast_to_json(G)

    def draw(self):
        """
        Renders and saves a visual representation of the graph.

        This method uses NetworkX and Graphviz to create a visual representation of the graph stored in
        self._G. It first exports the graph to a DOT file, then computes the layout for the graph using
        Graphviz"s "dot" program, and finally draws the graph. The resulting image is saved as "ast_graph.png"
        in the "output" directory. The method also clears the current figure after saving the image to prevent
        overlap with subsequent plots.

        Note:
            - This method assumes that the "output" directory exists.
            - The graph is saved without labels for clarity.
            - Graphviz must be installed and accessible in the system"s PATH for this method to work.
        """
        draw = Draw(None, None)
        draw.ast(self._G, self._t2t_paths)

    @staticmethod
    def from_notebook_windows(notebook: Notebook) -> List["PythonASTGraph"]:
        """
        Creates a list of ASTGraph objects from code cells of a given notebook.
        """
        ast_graphs = []
        for cell_window in notebook:
            G = PythonASTGraph()
            G.parse_code(notebook.cell_code(cell_window))
            ast_graphs.append(G)

        return ast_graphs
