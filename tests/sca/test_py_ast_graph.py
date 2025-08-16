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

    @patch.object(PythonASTGraph, 'add_node', return_value=0)
    @patch.object(PythonASTGraph, 'visit', side_effect=lambda x: 1 if isinstance(x, ast.AST) else x)
    def test_generic_visit(self, mock_visit, mock_add_node):
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
        self.py_ast_graph._G = MagicMock(spec=nx.DiGraph)

        self.py_ast_graph.generic_visit(node)

        mock_add_node.assert_called_once_with(node)
        # Check that visit is called for child nodes, but not for Load/Store
        mock_visit.assert_has_calls([
            call(node.value),
            call(node.value.left),
            call(node.value.op),
            call(node.value.right)
        ], any_order=True)
        
        # Check that edges are added
        self.py_ast_graph._G.add_edge.assert_has_calls([
            call(0, 1), # Assign -> BinOp
            call(1, 1), # BinOp -> Constant (left)
            call(1, 1), # BinOp -> Add
            call(1, 1)  # BinOp -> Constant (right)
        ], any_order=True)

    @patch.object(PythonASTGraph, 'get_code_from_node', return_value="mock_code")
    def test_add_node(self, mock_get_code_from_node):
        initial_node_counter = self.py_ast_graph.node_counter
        mock_node = MagicMock(spec=ast.AST)
        mock_node.__class__.__name__ = "MockNode"

        self.py_ast_graph._G = MagicMock(spec=nx.DiGraph)

        node_id = self.py_ast_graph.add_node(mock_node)

        self.assertEqual(node_id, initial_node_counter)
        self.assertEqual(self.py_ast_graph.node_counter, initial_node_counter + 1)
        self.py_ast_graph._G.add_node.assert_called_once_with(
            initial_node_counter, label="MockNode", code="mock_code"
        )
        mock_get_code_from_node.assert_called_once_with(mock_node)

    @patch('ApexDAG.util.draw.Draw')
    def test_draw(self, MockDraw):
        mock_draw_instance = MockDraw.return_value
        self.py_ast_graph.draw()
        MockDraw.assert_called_once_with(None, None)
        mock_draw_instance.ast.assert_called_once_with(self.py_ast_graph._G, self.py_ast_graph._t2t_paths)

    @patch('ApexDAG.sca.py_ast_graph.PythonASTGraph')
    def test_from_notebook_windows(self, MockPythonASTGraph):
        mock_notebook = MagicMock(spec=Notebook)
        mock_notebook.__iter__.return_value = ['cell1', 'cell2']
        mock_notebook.cell_code.side_effect = ['code1', 'code2']

        mock_graph_instance1 = MockPythonASTGraph.return_value
        mock_graph_instance2 = MockPythonASTGraph.return_value
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
