import ast
from abc import ABC
from copy import deepcopy

import networkx as nx
from tqdm import tqdm

from SystemX.notebook import Notebook

class ASTGraph(ABC):  # noqa: B024
    def __init__(self) -> None:
        """Initializes a new instance of the class."""
        self._G = nx.DiGraph()
        self._build = False
        self._t2t_paths = []
        self._leaf_nodes = []
        self.node_counter = 0

    def create_notebook_root(self) -> int | None:
        """Optional hook for subclasses to establish a master root node for the entire notebook."""
        return None

    def connect_notebook_root(self, root_id: int, cell_module_id: int) -> None:  # noqa: B027
        """Optional hook to attach a disconnected cell subgraph to the master notebook root."""

    def pre_parse(self, sources: list[str]) -> None:  # noqa: B027
        """Optional hook called once with every source that will be parsed, before any AST is visited."""

    def transform_tree(self, tree: ast.Module) -> ast.Module:
        """Optional hook to rewrite a freshly parsed AST before it is visited."""
        return tree

    def parse_code(self, code: str) -> None:
        """Parses the given source code string into an abstract syntax tree (AST) and visits the nodes."""
        self.code = code
        self.pre_parse([code])
        abstract_syntax_tree = ast.parse(code)
        abstract_syntax_tree = self.transform_tree(abstract_syntax_tree)
        self.visit(abstract_syntax_tree)
        self._build = True

    def parse_cells(self, cells: list) -> None:
        """Parses a list of Jupyter cell dictionaries into abstract syntax trees (ASTs)."""
        self.cells = cells
        self.code = ""

        self.pre_parse([cell.get("source", "") for cell in cells])

        self.current_cell_id = "global_notebook"
        self.current_cell_source = "<notebook_root>"
        master_root_id = self.create_notebook_root()

        for index, cell in enumerate(cells):
            self.current_cell_id = cell.get("cell_id") or cell.get("id") or f"cell_{index}"
            source = cell.get("source", "")

            self.current_cell_source = source
            self.code += source + "\n"

            if not source.strip():
                continue

            try:
                abstract_syntax_tree = ast.parse(source)
                abstract_syntax_tree = self.transform_tree(abstract_syntax_tree)
                cell_module_id = self.visit(abstract_syntax_tree)

                if master_root_id is not None and cell_module_id is not None:
                    self.connect_notebook_root(master_root_id, cell_module_id)

            except SyntaxError as e:
                raise SyntaxError(f"SyntaxError in cell {self.current_cell_id}: {e}") from e

        self._build = True
        self.current_cell_id = None

    def parse_notebook(self, notebook: Notebook) -> None:
        """Parses the source code contained in a given Notebook object."""
        self.parse_code(notebook.code())

    def get_graph(self) -> nx.DiGraph:
        """Retrieves a deep copy of the directed graph associated with this object."""
        self._check_graph_status()
        return deepcopy(self._G)

    def get_code(self) -> str:
        """Retrieves the source code associated with the graph."""
        self._check_graph_status()
        return self.code

    def get_code_from_node(self, node: object) -> str:
        """Retrieves the source code corresponding to a given AST (Abstract Syntax Tree) node."""
        source_to_use = getattr(self, "current_cell_source", None) or self.code

        lines = source_to_use.splitlines()
        if not lines or node.lineno > len(lines):
            return ""

        return lines[node.lineno - 1][node.col_offset : node.end_col_offset]

    def draw(self) -> None:  # noqa: B027
        """Optional hook: renders and saves a visual representation of the graph."""

    def get_nodes(self) -> list[nx.classes.reportviews.NodeView]:
        """Retrieves a list of all nodes present in the graph."""
        self._check_graph_status()
        return self._G.nodes

    def get_edges(self) -> list[nx.classes.reportviews.EdgeView]:
        """Retrieves a list of all edges present in the graph."""
        self._check_graph_status()
        return self._G.edges

    def get_leaf_nodes(self) -> list[nx.classes.reportviews.NodeView]:
        """Retrieves a list of leaf nodes present in the graph."""
        self._check_graph_status()
        return self._leaf_nodes

    def get_t2t_paths(self, max_depth: int = 5) -> list[list[int]]:
        """Retrieves all simple paths between leaf nodes in the graph up to a specified maximum depth."""
        self._check_graph_status()
        if self._t2t_paths:
            return self._t2t_paths

        self._leaf_nodes = [node for node in self._G.nodes if self._G.out_degree(node) == 0]

        G = self._G.to_undirected()
        leaf_to_leaf_paths = []
        for leaf_node in tqdm(self._leaf_nodes, desc="Processing leaf nodes"):
            for other_leaf_node in self._leaf_nodes:
                if leaf_node != other_leaf_node:
                    simple_paths = nx.all_simple_paths(G, leaf_node, other_leaf_node, cutoff=max_depth)
                    for path in simple_paths:
                        leaf_to_leaf_paths.append(path)

        self._t2t_paths = leaf_to_leaf_paths
        return self._t2t_paths

    def _check_graph_status(self) -> None:
        """Checks whether the graph has been built."""
        if not self._build:
            raise ASTGraphNotBuildError("Graph not built. Please parse a notebook or code first.")

class ASTGraphNotBuildError(RuntimeError):
    """Exception raised when an operation is attempted on an unbuilt graph."""

    pass
