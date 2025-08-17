import unittest
from unittest.mock import MagicMock, patch, call
from ApexDAG.util.notebook_stat_miner import NotebookStatMiner
from ApexDAG.notebook import Notebook
from ApexDAG.sca.import_visitor import ImportVisitor


class TestNotebookStatMiner(unittest.TestCase):
    def setUp(self):
        self.mock_notebook = MagicMock(spec=Notebook)
        self.miner = NotebookStatMiner(self.mock_notebook)

    def test_initialization(self):
        self.assertEqual(self.miner.notebook, self.mock_notebook)
        self.assertEqual(self.miner.imports, [])
        self.assertEqual(self.miner.classes, [])
        self.assertEqual(self.miner.functions, [])
        self.assertEqual(self.miner.import_usage, {})
        self.assertEqual(self.miner.import_counts, {})
        self.assertEqual(self.miner.cells_with_imports, 0)

    def test_convert_to_array_valid(self):
        self.assertEqual(NotebookStatMiner.convert_to_array("[1, 2, 3]"), [1, 2, 3])
        self.assertEqual(NotebookStatMiner.convert_to_array('["a", "b"]'), ["a", "b"])

    def test_convert_to_array_invalid(self):
        self.assertEqual(NotebookStatMiner.convert_to_array("abc"), [])
        self.assertEqual(NotebookStatMiner.convert_to_array(""), [])

    def test_process_results(self):
        mock_import_visitor = MagicMock(spec=ImportVisitor)
        mock_import_visitor.imports = [("os", None)]
        mock_import_visitor.classes = ["MyClass"]
        mock_import_visitor.functions = ["my_func"]

        self.miner._process_results(mock_import_visitor)

        self.assertEqual(self.miner.imports, [("os", None)])
        self.assertEqual(self.miner.classes, ["MyClass"])
        self.assertEqual(self.miner.functions, ["my_func"])
        self.assertEqual(self.miner.cells_with_imports, 1)

        # Test with no imports
        mock_import_visitor.imports = []
        self.miner._process_results(mock_import_visitor)
        self.assertEqual(self.miner.cells_with_imports, 1)  # Should not increment again

    @patch("ast.parse")
    @patch("ApexDAG.sca.import_visitor.ImportVisitor")
    def test_mine_import_usages(self, MockImportVisitor, mock_ast_parse):
        self.mock_notebook.code.return_value = "import os"
        mock_visitor_instance = MockImportVisitor.return_value
        mock_visitor_instance.import_usage = {"os": set()}
        mock_visitor_instance.import_counts = {"os": 1}

        self.miner._mine_import_usages()

        self.mock_notebook.code.assert_called_once()
        MockImportVisitor.assert_called_once()
        mock_ast_parse.assert_called_once_with("import os")
        mock_visitor_instance.visit.assert_called_once()
        self.assertEqual(self.miner.import_usage, {"os": set()})
        self.assertEqual(self.miner.import_counts, {"os": 1})

    @patch.object(NotebookStatMiner, "_process_results")
    @patch("ast.parse")
    @patch("ApexDAG.sca.import_visitor.ImportVisitor")
    def test_mine_cell_data(
        self, MockImportVisitor, mock_ast_parse, mock_process_results
    ):
        self.mock_notebook.__iter__.return_value = iter(
            [([{"code": "cell1_code"}],), ([{"code": "cell2_code"}],)]
        )
        mock_visitor_instance = MockImportVisitor.return_value

        self.miner._mine_cell_data()

        self.assertEqual(MockImportVisitor.call_count, 2)
        self.assertEqual(mock_ast_parse.call_count, 2)
        mock_ast_parse.assert_has_calls([call("cell1_code"), call("cell2_code")])
        self.assertEqual(mock_visitor_instance.visit.call_count, 2)
        self.assertEqual(mock_process_results.call_count, 2)
        mock_process_results.assert_has_calls(
            [call(mock_visitor_instance), call(mock_visitor_instance)]
        )

    @patch.object(NotebookStatMiner, "_mine_cell_data")
    @patch.object(NotebookStatMiner, "_mine_import_usages")
    def test_mine(self, mock_mine_import_usages, mock_mine_cell_data):
        self.mock_notebook._cell_window_size = 5  # Simulate initial window size
        self.mock_notebook.create_execution_graph = MagicMock()

        imports, usage, counts, classes, functions, cells_with_imports = (
            self.miner.mine(greedy=True)
        )

        self.assertEqual(self.mock_notebook._cell_window_size, 1)
        self.mock_notebook.create_execution_graph.assert_called_once_with(greedy=True)
        mock_mine_cell_data.assert_called_once()
        mock_mine_import_usages.assert_called_once()
        self.assertEqual(self.mock_notebook._cell_window_size, 5)  # Should be restored
        self.mock_notebook.create_execution_graph.assert_called_with(greedy=True)


if __name__ == "__main__":
    unittest.main()
