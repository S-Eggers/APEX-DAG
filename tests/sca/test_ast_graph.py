import unittest
from unittest.mock import MagicMock, patch
import networkx as nx
from ApexDAG.sca.ast_graph import ASTGraph, ASTGraphNotBuildError
from ApexDAG.notebook import Notebook


class TestASTGraph(unittest.TestCase):
    def setUp(self):
        # Create a concrete implementation for testing the abstract base class
        class ConcreteASTGraph(ASTGraph):
            def visit(self, node):
                pass  # No-op for testing abstract methods

            def draw(self):
                pass  # No-op for testing abstract methods

        self.ast_graph = ConcreteASTGraph()

    def test_ast_graph_not_build_error(self):
        self.assertTrue(issubclass(ASTGraphNotBuildError, RuntimeError))

    def test_initialization(self):
        self.assertIsInstance(self.ast_graph._G, nx.DiGraph)
        self.assertFalse(self.ast_graph._build)
        self.assertEqual(self.ast_graph._t2t_paths, [])
        self.assertEqual(self.ast_graph._leaf_nodes, [])
        self.assertEqual(self.ast_graph.node_counter, 0)

    def test_check_graph_status_not_built(self):
        self.ast_graph._build = False
        with self.assertRaises(ASTGraphNotBuildError):
            self.ast_graph._check_graph_status()

    def test_check_graph_status_built(self):
        self.ast_graph._build = True
        # Should not raise an error
        self.ast_graph._check_graph_status()

    @patch("ast.parse")
    def test_parse_code(self, mock_ast_parse):
        mock_ast_parse.return_value = MagicMock()
        code = "print('hello')"
        self.ast_graph.parse_code(code)
        self.assertEqual(self.ast_graph.code, code)
        mock_ast_parse.assert_called_once_with(code)
        self.assertTrue(self.ast_graph._build)

    @patch.object(ASTGraph, "parse_code")
    def test_parse_notebook(self, mock_parse_code):
        mock_notebook = MagicMock(spec=Notebook)
        mock_notebook.code.return_value = "notebook code"
        self.ast_graph.parse_notebook(mock_notebook)
        mock_notebook.code.assert_called_once()
        mock_parse_code.assert_called_once_with("notebook code")

    @patch.object(ASTGraph, "_check_graph_status")
    @patch("ApexDAG.sca.ast_graph.deepcopy")
    def test_get_graph(self, mock_deepcopy, mock_check_graph_status):
        mock_deepcopy.return_value = MagicMock()
        graph = self.ast_graph.get_graph()
        mock_check_graph_status.assert_called_once()
        mock_deepcopy.assert_called_once_with(self.ast_graph._G)
        self.assertEqual(graph, mock_deepcopy.return_value)

    def test_get_code_from_node(self):
        self.ast_graph.code = "line1\nline2_start_end\nline3"
        mock_node = MagicMock()
        mock_node.lineno = 2
        mock_node.col_offset = 11
        mock_node.end_col_offset = 15
        code_snippet = self.ast_graph.get_code_from_node(mock_node)
        self.assertEqual(code_snippet, "_end")

    def test_draw_abstract_method(self):
        # Test that TypeError is raised if draw is not implemented in a concrete class
        class IncompleteASTGraph(ASTGraph):
            def visit(self, node):
                pass

        with self.assertRaises(TypeError):
            IncompleteASTGraph()

    @patch.object(ASTGraph, "_check_graph_status")
    def test_get_nodes(self, mock_check_graph_status):
        self.ast_graph._G.add_node("a")
        nodes = self.ast_graph.get_nodes()
        mock_check_graph_status.assert_called_once()
        self.assertEqual(list(nodes), ["a"])

    @patch.object(ASTGraph, "_check_graph_status")
    def test_get_edges(self, mock_check_graph_status):
        self.ast_graph._G.add_edge("a", "b")
        edges = self.ast_graph.get_edges()
        mock_check_graph_status.assert_called_once()
        self.assertEqual(list(edges), [("a", "b")])

    @patch.object(ASTGraph, "_check_graph_status")
    def test_get_leaf_nodes(self, mock_check_graph_status):
        self.ast_graph._leaf_nodes = ["leaf1", "leaf2"]
        leaf_nodes = self.ast_graph.get_leaf_nodes()
        mock_check_graph_status.assert_called_once()
        self.assertEqual(leaf_nodes, ["leaf1", "leaf2"])

    @patch.object(ASTGraph, "_check_graph_status")
    @patch("networkx.all_simple_paths")
    @patch(
        "tqdm.tqdm", side_effect=lambda iterable, *args, **kwargs: iterable
    )  # Mock tqdm to return iterable directly
    def test_get_t2t_paths_calculate(
        self, mock_tqdm, mock_all_simple_paths, mock_check_graph_status
    ):
        self.ast_graph._G.add_edges_from([("a", "b"), ("b", "c"), ("d", "e")])
        # 'c' and 'e' are leaf nodes
        mock_all_simple_paths.side_effect = [
            iter([["c", "b", "a", "d", "e"]]),
            iter([]),
        ]

        paths = self.ast_graph.get_t2t_paths()

        mock_check_graph_status.assert_called()
        self.assertEqual(self.ast_graph._leaf_nodes, ["c", "e"])
        self.assertEqual(mock_all_simple_paths.call_count, 2)
        self.assertEqual(paths, [["c", "b", "a", "d", "e"]])
        self.assertEqual(self.ast_graph._t2t_paths, [["c", "b", "a", "d", "e"]])

    @patch.object(ASTGraph, "_check_graph_status")
    def test_get_t2t_paths_cached(self, mock_check_graph_status):
        self.ast_graph._t2t_paths = [["cached_path"]]
        paths = self.ast_graph.get_t2t_paths()
        mock_check_graph_status.assert_called_once()
        self.assertEqual(paths, [["cached_path"]])


if __name__ == "__main__":
    unittest.main()
