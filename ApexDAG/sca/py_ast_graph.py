import ast
from typing import List
from ApexDAG.notebook import Notebook

from ApexDAG.sca import ASTGraph
from ApexDAG.util.draw import Draw
from ApexDAG.sca.constants import AST_NODE_TYPES, AST_EDGE_TYPES


class PythonASTGraph(ASTGraph, ast.NodeVisitor):
    
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
                        self._G.add_edge(
                            node_id, child_id, 
                            edge_type=AST_EDGE_TYPES["AST_PARENT_CHILD"], 
                            label=field
                        )
            elif isinstance(value, ast.AST) and not isinstance(
                value, (ast.Load, ast.Store)
            ):
                child_id = self.visit(value)
                self._G.add_edge(
                    node_id, child_id, 
                    edge_type=AST_EDGE_TYPES["AST_PARENT_CHILD"], 
                    label=field
                )

        return node_id

    def add_node(self, node) -> int:
        """
        Adds a new node to the graph representing an AST node.
        """
        node_label = type(node).__name__
        code = ""
        if (
            hasattr(node, "lineno")
            and hasattr(node, "col_offset")
            and hasattr(node, "end_col_offset")
        ):
            code = self.get_code_from_node(node)

        numeric_type = AST_NODE_TYPES.get(node_label, AST_NODE_TYPES["AST_UNKNOWN"])

        self._G.add_node(
            self.node_counter, 
            label=node_label, 
            code=code, 
            node_type=numeric_type
        )
        
        node_id = self.node_counter
        self.node_counter += 1
        return node_id

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

        This static method iterates over code cells in the provided notebook. For each cell window, it
        creates a new ASTGraph object and parses the code in that cell into the graph. It then adds each
        of these ASTGraph objects to a list. This method is useful for analyzing or processing notebook
        code where each cell windows" code is considered as an independent entity.

        Args:
            notebook (Notebook): The notebook object containing the code cells.

        Returns:
            List[ASTGraph]: A list of ASTGraph objects, each representing the abstract syntax tree of a
            single code cell from the notebook.

        Note:
            - The Notebook class is expected to support iteration over its cells and provide a method
            `cell_code(cell_window)` to access the code in each cell.
            - This method is particularly useful when each cell"s code needs to be analyzed or processed
            separately rather than as a continuous script.
        """
        ast_graphs = []
        for cell_window in notebook:
            G = PythonASTGraph()
            G.parse_code(notebook.cell_code(cell_window))
            ast_graphs.append(G)

        return ast_graphs
