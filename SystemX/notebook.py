import re

import nbformat
import networkx as nx

from SystemX.util.logging import setup_logging

class Notebook:
    """A class for analyzing and processing Jupyter Notebooks."""

    VERBOSE = False

    def __init__(
        self,
        url: str,
        cell_window_size: int = 1,
        nb: nbformat.NotebookNode = None,
    ) -> None:
        """Initializes the object, loading a Jupyter Notebook from a specified URL or using an already loaded notebook object, and setting up a directed graph."""
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
        """Removes Jupyter Notebook specific lines from a given string of code."""
        pattern = r"^\s*(!|%|# In\[).*"

        return "\n".join(line for line in code.split("\n") if not re.match(pattern, line))

    def count_code_cells(self) -> int:
        """Returns the number of code cells in the notebook."""
        if self._exec_graph_exists:
            return len(self._G.nodes)

        return len([cell for cell in self._nb.cells if cell.cell_type == "code"])

    def path(self) -> str:
        """Returns the path of the notebook."""
        return self.url

    def create_execution_graph(self, greedy: bool = False) -> None:
        """Constructs an execution graph for a Jupyter Notebook's cells based on their execution order."""
        prev_index = -1
        node2execution_count = dict()
        for index, cell in enumerate(self._nb.cells):
            if "execution_count" in cell and cell.execution_count is not None:
                self._logger.debug(f"Adding cell {index} with execution count {cell.execution_count}")

                source = self.remove_jupyter_lines(cell.source)
                self._G.add_node(
                    index,
                    name=f"cell-{index}",
                    number=cell.execution_count,
                    code=source,
                )

                if greedy and prev_index >= 0:
                    self._G.add_edge(prev_index, index)
                else:
                    node2execution_count[index] = cell.execution_count

                if (cell.execution_count == 1 and not greedy) or self._source < 0:
                    self._source = index

                prev_index = index
            elif "execution_count" in cell and greedy:
                self._logger.debug(f"Adding cell {index} without execution count")

                source = self.remove_jupyter_lines(cell.source)
                self._G.add_node(index, name=f"cell-{index}", number=0, code=source)

                if prev_index >= 0:
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
        """Returns the code of the notebook as a single string."""
        try:
            node_indicies = [index for index in nx.dfs_preorder_nodes(self._G, self._source)]
            return "\n\n".join([self._G.nodes[i]["code"] for i in node_indicies])
        except nx.exception.NetworkXError as e:
            self._logger.warning(e)
            return ""

    def cell_code(self, indicies: list) -> str:
        """Returns the code of the notebook cells at the specified indicies as a single string."""
        return "\n".join([self._G.nodes[i]["code"] for i in indicies])

    def print_code(self) -> None:
        formatted_code = self.format_code()
        print(formatted_code)

    def save_code(self, file_path: str, mode: str = "w+", encoding: str = "utf-8") -> None:
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

    def __iter__(self) -> object:
        """Creates an iterator that iterates over windows of a specified size, based on the depth-first search (DFS) ordering of nodes in a graph."""
        if not self._exec_graph_exists:
            raise RuntimeError("No cell execution graph to iterate on")

        node_indicies = [index for index in nx.dfs_preorder_nodes(self._G, self._source)]
        if len(node_indicies) < self._cell_window_size and not self._cell_window_size == -1:
            raise ValueError("Cell window size is bigger than actual cells available. Reduce the window size.")

        if self._cell_window_size == -1:
            yield [self._G.nodes[i] for i in node_indicies]
        else:
            for i in range(len(node_indicies) - self._cell_window_size + 1):
                yield [self._G.nodes[j] for j in node_indicies[i : i + self._cell_window_size]]

    def __str__(self) -> str:
        """Returns a string representation of the Notebook object."""
        return f"Notebook: {self._nb.metadata.language_info.name}-{id(self)}\nNumber of cells: {len(self._nb.cells)}\nCell window size: {self._cell_window_size}"

    def __repr__(self) -> str:
        """Returns a string representation of the Notebook object."""
        return str(self)

    def __len__(self) -> int:
        """Returns the number of cells in the notebook."""
        return len(self._nb.cells)

    def __getitem__(self, index: int) -> object:
        """Returns the cell at the specified index."""
        return self._nb.cells[index]
