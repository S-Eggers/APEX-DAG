import unittest
from unittest.mock import MagicMock, patch, call
import ast
import networkx as nx
from ApexDAG.sca.py_ast_graph import PythonASTGraph
from ApexDAG.notebook import Notebook
from ApexDAG.util.draw import Draw

class TestPythonASTGraph(unittest.TestCase):

    def setUp(self):
        self.py_ast_graph = PythonASTGraph()
        self.py_ast_graph.code = "test_code"

    def test_inheritance(self):
        self.assertIsInstance(self.py_ast_graph, PythonASTGraph)
        self.assertIsInstance(self.py_ast_graph, ast.NodeVisitor)

    @patch.object(PythonASTGraph, 'add_node')
    def test_generic_visit(self, mock_add_node):
        # Create a side_effect that returns increasing integers
        node_id_counter = 0
        def mock_add_node_side_effect(node):
            nonlocal node_id_counter
            current_id = node_id_counter
            node_id_counter += 1
            return current_id

        mock_add_node.side_effect = mock_add_node_side_effect

        py_ast_graph = PythonASTGraph()
        # Create a simple AST structure for testing
        # Represents: a = 1 + 2
        node = ast.Assign(
            targets=[ast.Name(id='a', ctx=ast.Store())],
            value=ast.BinOp(
                left=ast.Constant(value=1),
                op=ast.Add(),
                right=ast.Constant(value=2)
            )
        )
        
        # Mock the internal graph to check edge additions
        py_ast_graph._G = MagicMock(spec=nx.DiGraph)

        # Call visit to trigger full traversal and add_node/add_edge calls
        py_ast_graph.visit(node)

        # Assertions for add_node calls
        # We need to map the nodes to their expected IDs
        node_to_id = {
            node: 0, # Assign
            node.targets[0]: 1, # Name
            node.value: 2, # BinOp
            node.value.left: 3, # Constant (left)
            node.value.op: 4, # Add
            node.value.right: 5 # Constant (right)
        }

        # Assert that add_node was called for all expected nodes
        expected_add_node_calls = [call(n) for n in node_to_id.keys()]
        mock_add_node.assert_has_calls(expected_add_node_calls, any_order=True)

        # Assertions for add_edge calls
        # Edges are from parent to child.
        # Assign (0) -> Name (1)
        # Assign (0) -> BinOp (2)
        # BinOp (2) -> Constant (3)
        # BinOp (2) -> Add (4)
        # BinOp (2) -> Constant (5)

        py_ast_graph._G.add_edge.assert_has_calls([
            call(node_to_id[node], node_to_id[node.targets[0]]),
            call(node_to_id[node], node_to_id[node.value]),
            call(node_to_id[node.value], node_to_id[node.value.left]),
            call(node_to_id[node.value], node_to_id[node.value.op]),
            call(node_to_id[node.value], node_to_id[node.value.right])
        ], any_order=True)

    @patch.object(PythonASTGraph, 'get_code_from_node', return_value="mock_code")
    def test_add_node(self, mock_get_code_from_node):
        initial_node_counter = self.py_ast_graph.node_counter
        mock_node = MagicMock(spec=ast.AST)
        mock_node.lineno = 1
        mock_node.col_offset = 0
        mock_node.end_col_offset = 10
        mock_node.__class__.__name__ = "MockNode"

        self.py_ast_graph._G = MagicMock(spec=nx.DiGraph)

        node_id = self.py_ast_graph.add_node(mock_node)

        self.assertEqual(node_id, initial_node_counter)
        self.assertEqual(self.py_ast_graph.node_counter, initial_node_counter + 1)
        self.py_ast_graph._G.add_node.assert_called_once_with(
            initial_node_counter, label="MagicMock", code="mock_code"
        )
        mock_get_code_from_node.assert_called_once_with(mock_node)

    @patch('ApexDAG.sca.py_ast_graph.Draw')
    def test_draw(self, MockDraw):
        py_ast_graph = PythonASTGraph()
        mock_draw_instance = MockDraw.return_value
        py_ast_graph.draw()
        MockDraw.assert_called_once_with(None, None)
        mock_draw_instance.ast.assert_called_once_with(py_ast_graph._G, py_ast_graph._t2t_paths)

    @patch('ApexDAG.sca.py_ast_graph.PythonASTGraph')
    def test_from_notebook_windows(self, MockPythonASTGraph):
        mock_notebook = MagicMock(spec=Notebook)
        mock_notebook.__iter__.return_value = ['cell1', 'cell2']
        mock_notebook.cell_code.side_effect = ['code1', 'code2']

        mock_graph_instance1 = MagicMock()
        mock_graph_instance2 = MagicMock()
        MockPythonASTGraph.side_effect = [mock_graph_instance1, mock_graph_instance2]

        ast_graphs = PythonASTGraph.from_notebook_windows(mock_notebook)

        self.assertEqual(len(ast_graphs), 2)
        self.assertEqual(ast_graphs[0], mock_graph_instance1)
        self.assertEqual(ast_graphs[1], mock_graph_instance2)

        mock_notebook.cell_code.assert_has_calls([
            call('cell1'),
            call('cell2')
        ])
        mock_graph_instance1.parse_code.assert_called_once_with('code1')
        mock_graph_instance2.parse_code.assert_called_once_with('code2')

if __name__ == '__main__':
    unittest.main()
