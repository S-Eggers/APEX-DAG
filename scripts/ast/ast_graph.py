import ast
import networkx as nx
from tqdm import tqdm
from typing import List
from copy import deepcopy
from abc import ABC, abstractmethod
from scripts.notebook import Notebook



class ASTGraph(ABC):
    def __init__(self):
        """
        Initializes a new instance of the class.

        This constructor initializes an empty directed graph using NetworkX, sets the build status to False, 
        and initializes empty lists for storing terminal-to-terminal paths and leaf nodes. It also initializes a node 
        counter to zero.

        Attributes:
            _G (nx.DiGraph): An empty directed graph instance created using the NetworkX library.
            _build (bool): A flag indicating whether a structure (e.g., a graph or tree) has been built. 
                        Initially set to False.
            _t2t_paths (list): A list to store paths between terminal nodes. Initially empty.
            _leaf_nodes (list): A list to store leaf nodes of a tree or graph. Initially empty.
            node_counter (int): A counter to keep track of the number of nodes. Initially set to zero.
        """
        self._G = nx.DiGraph()
        self._build = False
        self._t2t_paths = []
        self._leaf_nodes = []
        self.node_counter = 0
        
    def parse_code(self, code: str):
        """
        Parses the given source code string into an abstract syntax tree (AST) and visits the nodes.

        This method takes a string of source code, parses it into an AST using Python"s `ast` module, and 
        then visits each node of the AST. The visitation of nodes is handled by the `visit` method, which 
        should be implemented to define specific actions for different types of AST nodes. This method is 
        typically used to analyze or transform the source code by traversing its AST representation.

        Args:
            code (str): A string containing the source code to be parsed.

        Note:
            - The `visit` method used in this process should be defined elsewhere in the class and is 
            responsible for the actual processing of each AST node.
            - This method assumes that the input string is valid Python source code.
        """
        self.code = code
        abstract_syntax_tree = ast.parse(code)
        self.visit(abstract_syntax_tree)
        self._build = True
    
    def parse_notebook(self, notebook: Notebook):
        """
        Parses the source code contained in a given Notebook object.

        This method extracts the source code from the provided Notebook object and then parses this code 
        using the `parse_code` method of the same class. It is assumed that the Notebook object has a 
        method `code()` that returns its source code as a string. This method is useful for integrating 
        with environments where code is encapsulated in notebook formats, allowing the same parsing logic 
        to be applied as with raw source code strings.

        Args:
            notebook (Notebook): An object representing a notebook, which contains the source code to be parsed.

        Note:
            - The `Notebook` class should have a `code()` method that returns the source code as a string.
            - This method delegates the parsing process to the `parse_code` method, hence relies on its 
            implementation for parsing the code.
        """
        self.parse_code(notebook.code())
        pass
    
    def get_graph(self) -> nx.DiGraph:
        """
        Retrieves a deep copy of the directed graph associated with this object.

        This method returns a deep copy of the internal directed graph (of type nx.DiGraph from the NetworkX 
        library) that is maintained within the object. This ensures that the returned graph is a completely 
        separate instance from the internal graph, and modifications to the returned graph will not affect 
        the internal graph. This is particularly useful when the graph needs to be used or modified without 
        altering the original graph stored in the object.

        Returns:
            nx.DiGraph: A deep copy of the directed graph object representing the current state of the graph 
            within the object.

        Note:
            - The deep copy includes all nodes and edges, along with their attributes.
            - Since a deep copy is returned, modifications to the returned graph do not impact the internal graph.
        """
        self._check_graph_status()
        return deepcopy(self._G)
    
    def get_code_from_node(self, node) -> str:
        """
        Retrieves the source code corresponding to a given AST (Abstract Syntax Tree) node.

        This method takes an AST node and returns the source code corresponding to that node. It uses the
        node"s line and column offsets to extract the code from the source code string. The method assumes
        that the source code string is stored in the "code" attribute of the class (self.code).

        Args:
            node (ast.AST): The AST node whose source code is to be retrieved.

        Returns:
            str: The source code corresponding to the given AST node.

        Note:
            - This method assumes that the source code string is stored in the "code" attribute of the class.
            - The method assumes that the AST node has the following attributes: lineno, col_offset, and
            end_col_offset.
        """
        return self.code.splitlines()[node.lineno - 1][node.col_offset:node.end_col_offset]
    
    @abstractmethod
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
        pass
        
    def get_nodes(self) -> List[nx.classes.reportviews.NodeView]:
        """
        Retrieves a list of all nodes present in the graph.

        This method returns a view of all nodes in the graph maintained by the object, represented as a 
        list of NodeView objects from the NetworkX library. The NodeView provides access to the node"s 
        attributes and is useful for analyzing or manipulating the nodes in the graph. The list reflects 
        the current state of the nodes in the graph, including any nodes added or modified since the graph"s 
        creation.

        Returns:
            List[nx.classes.reportviews.NodeView]: A list-like view of the graph"s nodes, providing access 
            to each node"s attributes and identifiers.

        Note:
            - This method provides a read-only view of the nodes and does not allow for direct modification 
            of the graph.
            - The actual data structure returned is not a typical Python list but a NodeView object that 
            behaves like a list.
        """
        self._check_graph_status()
        return self._G.nodes
    
    def get_edges(self) -> List[nx.classes.reportviews.EdgeView]:
        """
        Retrieves a list of all edges present in the graph.

        This method returns a view of all edges in the graph maintained by the object, represented as a 
        list of EdgeView objects from the NetworkX library. The EdgeView provides access to the edge"s 
        attributes and endpoints, and is useful for analyzing or manipulating the edges in the graph. The 
        list reflects the current state of the edges in the graph, including any edges added or modified 
        since the graph"s creation.

        Returns:
            List[nx.classes.reportviews.EdgeView]: A list-like view of the graph"s edges, providing access 
            to each edge"s attributes and endpoints.

        Note:
            - This method provides a read-only view of the edges and does not allow for direct modification 
            of the graph.
            - The actual data structure returned is not a typical Python list but an EdgeView object that 
            behaves like a list.
        """
        self._check_graph_status()
        return self._G.edges
    
    def get_leaf_nodes(self) -> List[nx.classes.reportviews.NodeView]:
        """
        Retrieves a list of leaf nodes present in the graph.

        This method returns a list of nodes that are considered "leaf nodes" in the graph. A leaf node is 
        defined as a node with no outgoing edges. The nodes are represented as NodeView objects from the 
        NetworkX library, which provide access to the node"s attributes. This method is particularly useful 
        for scenarios where operations or analyses are focused on the leaf nodes of the graph.

        Returns:
            List[nx.classes.reportviews.NodeView]: A list of NodeView objects representing the leaf nodes 
            in the graph.

        Note:
            - This list is a snapshot of the leaf nodes at the time the method is called. Any modifications 
            to the graph after this call may not be reflected in the returned list.
            - The method assumes that the list of leaf nodes (`self._leaf_nodes`) is maintained and updated 
            appropriately as the graph is modified.
        """
        self._check_graph_status()
        return self._leaf_nodes
    
    def get_t2t_paths(self, max_depth=5) -> List[List[int]]:
        """
        Retrieves all simple paths between leaf nodes in the graph up to a specified maximum depth.

        This method computes and returns a list of all simple paths between leaf nodes of the graph. A leaf 
        node is defined as a node with no outgoing edges. Paths are computed in the undirected version of 
        the graph and are restricted to the specified maximum depth to limit their length. If the list of 
        paths has already been computed and stored in `self._t2t_paths`, this stored list is returned 
        directly. Otherwise, the method calculates these paths, stores them, and then returns them.

        Args:
            max_depth (int, optional): The maximum depth (or length) of the paths to be considered. 
            Defaults to 5.

        Returns:
            List[List[int]]: A list of lists, where each inner list is a sequence of node identifiers 
            representing a path between two leaf nodes.

        Note:
            - The method first converts the directed graph to an undirected graph to find simple paths 
            between nodes.
            - The method assumes that the graph (`self._G`) is up-to-date with all the nodes and edges 
            added so far.
            - Computation of paths might be resource-intensive for large graphs or high values of 
            `max_depth`.
        """
        self._check_graph_status()
        if self._t2t_paths:
            return self._t2t_paths
        
        self._leaf_nodes = [node for node in self._G.nodes if self._G.out_degree(node) == 0]
        
        # there are no terminal to terminal paths in a tree
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
    
    def _check_graph_status(self):
        """
        Checks whether the graph has been built.

        This method verifies if the graph associated with the object has been built or not. If the graph 
        has not been built (indicated by the "_build" attribute being False), the method raises an 
        exception. This check is useful to ensure that the graph is in a ready state before performing 
        operations that require a built graph, such as traversals, analyses, or visualizations.

        Raises:
            ASTGraphNotBuildError: If the graph has not been built, indicating that no source code or notebook 
            has been parsed to construct the graph.

        Note:
            - The graph is considered "built" if a notebook or source code has been successfully parsed and 
            the corresponding AST has been processed to construct the graph.
            - This method should be called before any operation that requires a complete and constructed graph.
        """
        if not self._build:
            raise ASTGraphNotBuildError("Graph not built. Please parse a notebook or code first.")


class ASTGraphNotBuildError(RuntimeError):
    """Exception raised when an operation is attempted on an unbuilt graph."""
    pass