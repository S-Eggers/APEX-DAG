import unittest
from unittest.mock import MagicMock, patch, call, ANY, mock_open
import ast
import networkx as nx
import numpy as np
import random

from ApexDAG.vamsa.generate_wir import (
    extract_from_node,
    GenPR,
    remove_assignments,
    filter_PRs,
    fix_bibartie_issue_import_from,
    construct_bipartite_graph,
    draw_graph,
    GenWIR
)
from ApexDAG.vamsa.utils import WIRNode, PRType

class TestGenerateWIR(unittest.TestCase):

    def setUp(self):
        # Reset seeds for consistent testing
        random.seed(42)
        np.random.seed(42)

    @patch('ApexDAG.vamsa.generate_wir.add_id', side_effect=lambda: ':id_mock')
    def test_extract_from_node_assign(self, mock_add_id):
        node = MagicMock(spec=ast.Assign)
        node.__class__.__name__ = "Assign"
        node.value = MagicMock(spec=ast.Constant)
        node.targets = [MagicMock(spec=ast.Name)]

        op = extract_from_node(WIRNode(node), "operation")
        self.assertEqual(op.node, "Assign:id_mock")
        self.assertFalse(op.isAttribute)

        input_node = extract_from_node(WIRNode(node), "input")
        self.assertEqual(input_node.node, node.value)

        output_node = extract_from_node(WIRNode(node), "output")
        self.assertEqual(output_node.node, node.targets)

    @patch('ApexDAG.vamsa.generate_wir.add_id', side_effect=lambda: ':id_mock')
    def test_extract_from_node_call(self, mock_add_id):
        node = MagicMock(spec=ast.Call)
        node.__class__.__name__ = "Call"
        node.func = MagicMock(spec=ast.Name, id="func_name")
        node.args = [MagicMock(spec=ast.Constant)]
        node.keywords = []

        op = extract_from_node(WIRNode(node), "operation")
        self.assertEqual(op.node, node.func)
        self.assertTrue(op.isAttribute)

        input_node = extract_from_node(WIRNode(node), "input")
        self.assertEqual(input_node.node, node.args + node.keywords)

        output_node = extract_from_node(WIRNode(node), "output")
        self.assertEqual(output_node.node, "func_name:id_mock")

    @patch('ApexDAG.vamsa.generate_wir.add_id', side_effect=lambda: ':id_mock')
    @patch('ApexDAG.vamsa.generate_wir.flatten', side_effect=lambda x: x)
    @patch('ApexDAG.vamsa.generate_wir.extract_from_node', side_effect=lambda n, f: WIRNode(f"mock_{f}"))
    def test_gen_pr_simple(self, mock_extract_from_node, mock_flatten, mock_add_id):
        node = MagicMock(spec=ast.Assign)
        node.__class__.__name__ = "Assign"
        
        result_o, result_prs = GenPR(WIRNode(node), [])

        self.assertIsInstance(result_o, WIRNode)
        self.assertIsInstance(result_prs, list)
        self.assertEqual(len(result_prs), 1)
        self.assertEqual(result_prs[0], ('mock_input', 'mock_caller', 'mock_operation', 'mock_output'))

    @patch('ApexDAG.vamsa.generate_wir.remove_id', side_effect=lambda x: x.split(':')[0] if x and ':' in x else x)
    def test_remove_assignments(self, mock_remove_id):
        prs = [
            ('in1', None, 'Assign:id1', 'out1'),
            ('in2', 'c2', 'op2', 'out1'),
            ('in3', 'c3', 'op3', 'out3'),
            ('out1', 'c4', 'op4', 'out4')
        ]
        filtered_prs = remove_assignments(prs)
        self.assertEqual(len(filtered_prs), 2)
        self.assertIn(('in2', 'c2', 'op2', 'out4'), filtered_prs)
        self.assertIn(('in3', 'c3', 'op3', 'out3'), filtered_prs)

    @patch('ApexDAG.vamsa.generate_wir.remove_id', side_effect=lambda x: x.split(':')[0] if x and ':' in x else x)
    def test_filter_prs(self, mock_remove_id):
        prs = [
            ('in1', 'c1', 'op1:id1', 'op1:id1'), # problematic
            ('in2', None, 'op1:id1', 'out2'), # should be fixed
            ('in3', 'c3', 'op3', 'out3')
        ]
        filtered_prs = filter_PRs(prs)
        self.assertEqual(len(filtered_prs), 2)
        self.assertIn(('in2', 'c1', 'op1:id1', 'out2'), filtered_prs)
        self.assertIn(('in3', 'c3', 'op3', 'out3'), filtered_prs)

    @patch('ApexDAG.vamsa.generate_wir.add_id', side_effect=lambda: ':id_mock')
    @patch('ApexDAG.vamsa.generate_wir.remove_id', side_effect=lambda x: x.split(':')[0] if x and ':' in x else x)
    def test_fix_bibartie_issue_import_from(self, mock_remove_id, mock_add_id):
        prs = [
            ('module', None, 'ImportFrom:id1', 'name:id2'),
            ('name:id2', 'c2', 'op2', 'out2')
        ]
        fixed_prs = fix_bibartie_issue_import_from(prs)
        self.assertEqual(len(fixed_prs), 3)
        self.assertIn(('module', None, 'ImportFrom:id1', 'name:id2:id_mock'), fixed_prs)
        self.assertIn((None, 'name:id2:id_mock', 'name:id2', None), fixed_prs)
        self.assertIn(('name:id2:id_mock', 'c2', 'op2', 'out2'), fixed_prs)

    @patch('ApexDAG.vamsa.generate_wir.remove_id', side_effect=lambda x: x.split(':')[0] if x and ':' in x else x)
    @patch('ApexDAG.vamsa.generate_wir.draw_graph')
    def test_construct_bipartite_graph(self, mock_draw_graph, mock_remove_id):
        prs = [
            ('in1', 'c1', 'op1', 'out1'),
            ('in2', None, 'op2', 'out2')
        ]
        graph, (inputs, outputs, callers, operations) = construct_bipartite_graph(prs, "output.png", True)

        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(len(graph.nodes()), 6) # in1, c1, op1, out1, in2, op2, out2
        self.assertEqual(len(graph.edges()), 5)
        self.assertIn('in1', inputs)
        self.assertIn('out1', outputs)
        self.assertIn('c1', callers)
        self.assertIn('op1', operations)
        mock_draw_graph.assert_called_once()

    @patch('matplotlib.pyplot.figure')
    @patch('networkx.drawing.nx_agraph.graphviz_layout', return_value={})
    @patch('networkx.draw_networkx_nodes')
    @patch('networkx.draw_networkx_edges')
    @patch('networkx.draw_networkx_labels')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    @patch('ApexDAG.vamsa.generate_wir.remove_id', side_effect=lambda x: x)
    def test_draw_graph(self, mock_remove_id, mock_close, mock_savefig, mock_legend, mock_labels, mock_edges, mock_nodes, mock_layout, mock_figure):
        graph = nx.DiGraph()
        graph.add_node('node1')
        draw_graph(graph, {'node1'}, set(), set(), set(), "test.png")
        mock_figure.assert_called_once()
        mock_layout.assert_called_once()
        mock_nodes.assert_called_once()
        mock_edges.assert_called_once()
        mock_labels.assert_called_once()
        mock_legend.assert_called_once()
        mock_savefig.assert_called_once_with("test.png")
        mock_close.assert_called_once()

    @patch('ApexDAG.vamsa.generate_wir.GenPR', side_effect=lambda n, p: (WIRNode(f"mock_output_{n.node.__class__.__name__}"), p + [('mock_in', 'mock_c', f'mock_op_{n.node.__class__.__name__}', 'mock_out')]))
    @patch('ApexDAG.vamsa.generate_wir.merge_prs', side_effect=lambda p1, p2: p1 + p2)
    @patch('ApexDAG.vamsa.generate_wir.filter_PRs', side_effect=lambda p: p)
    @patch('ApexDAG.vamsa.generate_wir.fix_bibartie_issue_import_from', side_effect=lambda p: p)
    @patch('ApexDAG.vamsa.generate_wir.remove_assignments', side_effect=lambda p: p)
    @patch('ApexDAG.vamsa.generate_wir.check_bipartie', return_value=True)
    @patch('ApexDAG.vamsa.generate_wir.construct_bipartite_graph', return_value=(nx.DiGraph(), (set(), set(), set(), set())))
    @patch('ast.iter_child_nodes', return_value=[MagicMock(spec=ast.Assign), MagicMock(spec=ast.Call)])
    @patch('builtins.open', new_callable=mock_open)
    @patch('ApexDAG.vamsa.generate_wir.logger.warning')
    def test_gen_wir(self, mock_logger_warning, mock_open_file, mock_construct_bipartite_graph, mock_check_bipartie, mock_remove_assignments, mock_fix_bipartite, mock_filter_prs, mock_merge_prs, mock_gen_pr, mock_iter_child_nodes):
        mock_assign_node = MagicMock(spec=ast.Assign)
        mock_assign_node.__class__.__name__ = "Assign"
        mock_call_node = MagicMock(spec=ast.Call)
        mock_call_node.__class__.__name__ = "Call"
        mock_iter_child_nodes.return_value = [mock_assign_node, mock_call_node]

        wir_graph, prs, node_sets = GenWIR(MagicMock(spec=ast.Module), "output.png", True)

        mock_gen_pr.assert_has_calls([
            call(WIRNode(mock_assign_node), []),
            call(WIRNode(mock_call_node), ANY) # PRs will be updated
        ])
        mock_merge_prs.call_count == 2
        mock_filter_prs.assert_called_once()
        mock_fix_bipartite.assert_called_once()
        mock_remove_assignments.assert_called_once()
        mock_check_bipartie.assert_called_once()
        mock_construct_bipartite_graph.assert_called_once_with(ANY, "output.png", True)
        mock_open_file.assert_called_once_with('output.txt', 'w')
        mock_logger_warning.assert_called_once()

if __name__ == '__main__':
    unittest.main()