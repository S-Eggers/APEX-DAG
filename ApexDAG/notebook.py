import re
import nbformat
import networkx as nx

from ApexDAG.util.logging import setup_logging


class Notebook:
    """
    A class for analyzing and processing Jupyter Notebooks.

    This class provides functionalities for reading a Jupyter Notebook, removing specific Jupyter lines from the 
    code, constructing an execution graph based on cell execution order, and iterating over the graph in windowed segments.

    The execution graph is a directed graph where each node represents a notebook cell, and edges represent the 
    execution order of these cells. The class supports two modes of graph construction: a greedy mode that connects
    each cell sequentially, and a non-greedy mode that connects cells based on their actual execution order.

    Attributes:
        _nb (NotebookNode): The Jupyter Notebook loaded from the specified URL.
        _G (DiGraph): A directed graph representing the cell execution flow within the notebook.
        _source (int): The starting point of the execution graph, typically the first executed cell.
        _exec_graph_exists (bool): Indicates whether the execution graph has been created.
        _cell_window_size (int): The size of the window for analyzing cell execution in the graph.

    Methods:
        __init__(url, cell_window_size, nb): Initializes the Notebook object.
        remove_jupyter_lines(code): Static method to remove Jupyter-specific lines from a given string of code.
        create_execution_graph(greedy): Constructs the execution graph for the notebook.
        code(): Returns the code of the notebook as a single string.
        __iter__(): Creates an iterator for traversing the execution graph in windowed segments.
        __str__(): Returns a string representation of the Notebook object.
        __repr__(): Returns a string representation of the Notebook object.
        __len__(): Returns the number of cells in the notebook.
        __getitem__(index): Returns the cell at the specified index.

    Example:
        >>> notebook = Notebook("notebook.ipynb", cell_window_size=2)
        >>> notebook.create_execution_graph(greedy=False)
        >>> for window in notebook:
        >>>     print(window)
    """
    VERBOSE = False

    def __init__(self, url: str, cell_window_size: int = 1, nb: nbformat.NotebookNode = None,):
        """
        Initializes the object, loading a Jupyter Notebook from a specified URL or using an already loaded notebook object,
        and setting up a directed graph.

        This constructor reads a Jupyter Notebook from the given URL using nbformat or uses the provided notebook object,
        and initializes a directed graph (DiGraph) using networkx to represent the notebook's cell execution flow. 
        It sets up various attributes for managing the notebook and its execution graph.

        Args:
            url (str, optional): The URL from which to load the Jupyter Notebook. The notebook is expected to be in a format
                                 compatible with nbformat.
            nb (NotebookNode, optional): An already loaded Jupyter Notebook object.
            cell_window_size (int, optional): The size of the window for cell execution analysis. This size determines
                                              how many cells are considered together during the graph creation process.
                                              Defaults to 1, which means each cell is considered individually.

        Attributes:
            _nb: The Jupyter Notebook loaded from the specified URL or provided directly.
            _G: A directed graph (DiGraph) representing the cell execution flow within the notebook.
            _source: An integer used to track the starting point of the execution graph. Initialized to -1.
            _exec_graph_exists: A boolean flag indicating whether the execution graph has been created. Initialized to False.
            _cell_window_size: An integer representing the window size for cell execution analysis.

        Example:
            >>> notebook = Notebook("notebook.ipynb", cell_window_size=2)
        """
        if nb is not None:
            self._nb = nb
        elif url is not None:
            self._nb = nbformat.read(url, as_version=4)
        else:
            raise ValueError("Either 'url' or 'nb' must be provided.")

        self._G = nx.DiGraph()
        self._source = -1
        self._exec_graph_exists = False
        self._cell_window_size = cell_window_size
        self.url = url
        self._logger = setup_logging("notebook", verbose=self.VERBOSE)
        
    @staticmethod
    def remove_jupyter_lines(code: str) -> str:
        """
        Removes Jupyter Notebook specific lines from a given string of code.

        This static method processes a multiline string, which represents code, 
        and removes lines that are specific to Jupyter Notebooks. These include lines 
        starting with '!', '%', or '# In[' which are typically used for shell commands, 
        magic commands, and input cell indicators in Jupyter Notebooks.

        Args:
            code (str): A multiline string containing the code from which Jupyter 
                        specific lines are to be removed.

        Returns:
            str: The processed code with Jupyter specific lines removed. This is 
                returned as a single string, with lines separated by newline characters.
        """
        pattern = r'^\s*(!|%|# In\[).*'
        
        return "\n".join(
            line for line in code.split("\n")
            if not re.match(pattern, line)
        )
        
    def count_code_cells(self) -> int:
        """
        Returns the number of code cells in the notebook.

        Returns:
            int: The number of code cells in the notebook.
        """
        if self._exec_graph_exists:
            return len(self._G.nodes)

        return len([cell for cell in self._nb.cells if cell.cell_type == "code"])
    
    def path(self) -> str:
        """
        Returns the path of the notebook.

        Returns:
            str: The path of the notebook.
        """
        return self.url
    
    def create_execution_graph(self, greedy: bool = False):
        """
        Constructs an execution graph for a Jupyter Notebook's cells based on their execution order.

        This method iterates through the cells of a Jupyter Notebook and constructs a graph where each node
        represents a cell. Nodes are created for cells with a defined 'execution_count', and they are connected
        based on either a greedy approach or the actual execution order.

        In greedy mode, each cell is connected to the next one sequentially. In non-greedy mode, cells are 
        connected based on their execution count, indicating the actual order in which cells were executed in 
        the notebook.

        Args:
            greedy (bool): If True, the graph will be constructed by sequentially connecting each cell to the 
                        next one. If False, cells will be connected based on their execution order. Defaults to False.
        """
        prev_index = -1
        node2execution_count = dict()
        for index, cell in enumerate(self._nb.cells):
            if "execution_count" in cell and cell.execution_count is not None:
                self._logger.debug(f"Adding cell {index} with execution count {cell.execution_count}")
                
                source = self.remove_jupyter_lines(cell.source)
                self._G.add_node(index, name=f"cell-{index}", number=cell.execution_count, code=source)
                
                if greedy and prev_index >= 0:
                    # print(f"adding edge ({prev_index}, {index})")
                    self._G.add_edge(prev_index, index)
                else:
                    node2execution_count[index] = cell.execution_count
                
                if cell.execution_count == 1 and not greedy:
                    self._source = index
                elif self._source < 0:
                    self._source = index
                
                prev_index = index
            elif "execution_count" in cell and greedy:
                self._logger.debug(f"Adding cell {index} without execution count")
                
                source = self.remove_jupyter_lines(cell.source)
                self._G.add_node(index, name=f"cell-{index}", number=0, code=source)
                
                if prev_index >= 0:
                    # print(f"adding edge ({prev_index}, {index})")
                    self._G.add_edge(prev_index, index)
                
                if self._source < 0:
                    self._source = index
                
                prev_index = index
        
        if not greedy:
            self._logger.debug("Non-greedy mode: creating execution graph based on execution count")
            prev_index = -1
            for index, _ in sorted(node2execution_count.items(), key=lambda x: x[1]):
                if index < 0:
                    continue
                
                self._G.add_edge(prev_index, index)
                prev_index = index
        
        self._exec_graph_exists = True
    
    def code(self) -> str:
        """
        Returns the code of the notebook as a single string.

        Returns:
            str: The code of the notebook as a single string.
        """
        node_indicies = [index for index in nx.dfs_preorder_nodes(self._G, self._source)]
        return "\n\n".join([self._G.nodes[i]["code"] for i in node_indicies])
    
    def cell_code(self, indicies: list) -> str:
        """
        Returns the code of the notebook cells at the specified indicies as a single string.

        Args:
            indicies (list): A list of integers representing the indicies of the cells to return.

        Returns:
            str: The code of the notebook cells at the specified indicies as a single string.
        """
        return "\n".join([self._G.nodes[i]["code"] for i in indicies])
    
    def print_code(self):
        formatted_code = self.format_code()
        print(formatted_code)
        
    def save_code(self, file_path: str, mode: str="w+", encoding: str="utf-8"):
        formatted_code = self.format_code()

        with open(file_path, mode, encoding=encoding) as file:
            file.write(formatted_code)
        
    def format_code(self) -> str:
        code = self.code()
        code_lines = code.split("\n")
        
        formatted_code = "\n"
        for index, code_line in enumerate(code_lines):
            formatted_code += f"{index + 1}\t{code_line}\n"
        
        return formatted_code
    
    def loc(self) -> int:
        code = self.code()
        code_lines = code.split("\n")
        count_loc = 0
        for code_line in code_lines:
            clean_line = code_line.strip()
            if len(clean_line) > 0 and not clean_line.startswith(("#", "%", "!")):
                count_loc += 1
        return count_loc
    
    def __iter__(self):
        """
        Creates an iterator that iterates over windows of a specified size, 
        based on the depth-first search (DFS) ordering of nodes in a graph.

        This iterator traverses the nodes of the graph `self._G`, starting at the node `self._source`, 
        in the order determined by a depth-first search. For each window of the specified size 
        `self._cell_window_size`, a subset of nodes is returned as a list. Each window includes 
        consecutive nodes in the DFS order.

        Raises:
            RuntimeError: If no execution graph exists to iterate on.
            ValueError: If the window size is larger than the number of available nodes in the graph.

        Yields:
            list: A list of nodes (or their data), where each element represents a window. Each window 
            contains nodes from `self._G` as per the depth-first search order and the specified window size.
        """
        if not self._exec_graph_exists:
            raise RuntimeError("No cell execution graph to iterate on")

        node_indicies = [index for index in nx.dfs_preorder_nodes(self._G, self._source)]
        if len(node_indicies) < self._cell_window_size and not self._cell_window_size == -1:
            raise ValueError("Cell window size is bigger than actual cells available. Reduce the window size.")
        
        # yield all nodes if window is -1
        if self._cell_window_size == -1:
            yield [self._G.nodes[i] for i in node_indicies]
        else:
            for i in range(len(node_indicies) - self._cell_window_size + 1):
                yield [self._G.nodes[j] for j in node_indicies[i:i + self._cell_window_size]]
                
    def __str__(self):
        """
        Returns a string representation of the Notebook object.

        Returns:
            str: A string representation of the Notebook object, including the notebook name, the number of cells, 
                and the size of the cell window for analysis.
        """
        return f"Notebook: {self._nb.metadata.language_info.name}-{id(self)}\nNumber of cells: {len(self._nb.cells)}\nCell window size: {self._cell_window_size}"
    
    def __repr__(self):
        """
        Returns a string representation of the Notebook object.

        Returns:
            str: A string representation of the Notebook object, including the notebook name, the number of cells, 
                and the size of the cell window for analysis.
        """
        return str(self)
    
    def __len__(self):
        """
        Returns the number of cells in the notebook.

        Returns:
            int: The number of cells in the notebook.
        """
        return len(self._nb.cells)
    
    def __getitem__(self, index: int):
        """
        Returns the cell at the specified index.

        Args:
            index (int): The index of the cell to return.

        Returns:
            NotebookNode: The cell at the specified index.
        """
        return self._nb.cells[index]