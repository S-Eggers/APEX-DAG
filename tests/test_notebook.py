
import unittest
import nbformat
import networkx as nx
from ApexDAG.notebook import Notebook

class TestNotebook(unittest.TestCase):

    def setUp(self):
        self.notebook_obj = nbformat.v4.new_notebook(
            cells=[
                nbformat.v4.new_code_cell("print('hello')", execution_count=1),
                nbformat.v4.new_code_cell("a = 1", execution_count=2),
                nbformat.v4.new_code_cell("b = 2", execution_count=3),
                nbformat.v4.new_code_cell("# %matplotlib inline", execution_count=4)
            ]
        )
        self.notebook = Notebook(url=None, nb=self.notebook_obj)

    def test_init_with_nb(self):
        self.assertIsNotNone(self.notebook._nb)
        self.assertIsInstance(self.notebook._G, nx.DiGraph)

    def test_init_with_url(self):
        with open("test.ipynb", "w") as f:
            nbformat.write(self.notebook_obj, f)
        notebook = Notebook(url="test.ipynb")
        self.assertIsNotNone(notebook._nb)

    def test_remove_jupyter_lines(self):
        code = "!pip install pandas\n%matplotlib inline\n# In[1]:\nprint('hello')"
        cleaned_code = Notebook.remove_jupyter_lines(code)
        self.assertEqual(cleaned_code, "print('hello')")

    def test_create_execution_graph_greedy(self):
        self.notebook.create_execution_graph(greedy=True)
        self.assertTrue(self.notebook._exec_graph_exists)
        self.assertEqual(len(self.notebook._G.nodes), 4)
        self.assertEqual(len(self.notebook._G.edges), 3)

    def test_create_execution_graph_non_greedy(self):
        self.notebook.create_execution_graph(greedy=False)
        self.assertTrue(self.notebook._exec_graph_exists)
        self.assertEqual(len(self.notebook._G.nodes), 5)
        self.assertEqual(len(self.notebook._G.edges), 4)

    def test_code(self):
        self.notebook.create_execution_graph(greedy=True)
        code = self.notebook.code()
        self.assertIn("print('hello')", code)
        self.assertIn("a = 1", code)
        self.assertIn("b = 2", code)

    def test_iter(self):
        self.notebook.create_execution_graph(greedy=True)
        windows = list(self.notebook)
        self.assertEqual(len(windows), 4)

    def test_len(self):
        self.assertEqual(len(self.notebook), 4)

    def test_getitem(self):
        cell = self.notebook[0]
        self.assertEqual(cell.source, "print('hello')")

if __name__ == '__main__':
    unittest.main()
