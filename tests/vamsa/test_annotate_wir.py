import unittest
from unittest.mock import MagicMock, patch, call
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from ApexDAG.vamsa.annotate_wir import add_to_stack, extend_stack, KB, AnnotationWIR
from ApexDAG.vamsa.utils import remove_id

class TestVamsaAnnotateWIR(unittest.TestCase):

    def test_add_to_stack(self):
        stack = []
        add_to_stack(stack, 1)
        self.assertEqual(stack, [1])
        add_to_stack(stack, 1) # Should not add duplicate
        self.assertEqual(stack, [1])
        add_to_stack(stack, 2)
        self.assertEqual(stack, [1, 2])

    def test_extend_stack(self):
        stack = [1]
        extend_stack(stack, [2, 1, 3])
        self.assertEqual(stack, [1, 2, 3])

    def test_kb_init(self):
        kb = KB()
        self.assertIsInstance(kb.knowledge_base, pd.DataFrame)
        self.assertIsInstance(kb.knowledge_base_traversal, pd.DataFrame)
        self.assertFalse(kb.knowledge_base.empty)
        self.assertFalse(kb.knowledge_base_traversal.empty)

    @patch('ApexDAG.vamsa.annotate_wir.remove_id', side_effect=lambda x: x)
    def test_kb_call_all_params(self, mock_remove_id):
        kb = KB()
        inputs, outputs = kb("pandas", None, "data", "drop")
        self.assertEqual(inputs, ["features"])
        self.assertEqual(outputs, ["features"])

    @patch('ApexDAG.vamsa.annotate_wir.remove_id', side_effect=lambda x: x)
    def test_kb_call_some_params(self, mock_remove_id):
        kb = KB()
        inputs, outputs = kb(L="sklearn", p="LogisticRegression")
        self.assertEqual(inputs, ["features", "labels"])
        self.assertEqual(outputs, ["model"])

    @patch('ApexDAG.vamsa.annotate_wir.remove_id', side_effect=lambda x: x)
    def test_kb_call_no_match(self, mock_remove_id):
        kb = KB()
        inputs, outputs = kb("non_existent", None, None, None)
        self.assertEqual(inputs, [])
        self.assertEqual(outputs, [])

    @patch('ApexDAG.vamsa.annotate_wir.remove_id', side_effect=lambda x: x)
    def test_kb_call_multiple_matches(self, mock_remove_id):
        kb = KB()
        # Add a duplicate entry to force multiple matches
        kb.knowledge_base = pd.concat([
            kb.knowledge_base,
            pd.DataFrame([{"Library": "pandas", "Module": None, "Caller": None, "API Name": "read_csv", "Inputs": ["file_path"], "Outputs": ["data"]}]),
        ])
        with self.assertRaises(ValueError):
            kb(L="pandas", p="read_csv")

    @patch('ApexDAG.vamsa.annotate_wir.remove_id', side_effect=lambda x: x)
    def test_kb_back_query(self, mock_remove_id):
        kb = KB()
        inputs = kb.back_query(O=["model"], p="LogisticRegression")
        self.assertEqual(inputs, ["features", "labels"])

    @patch('ApexDAG.vamsa.annotate_wir.remove_id', side_effect=lambda x: x)
    def test_kb_back_query_no_match(self, mock_remove_id):
        kb = KB()
        inputs = kb.back_query(O=["non_existent"], p="non_existent")
        self.assertEqual(inputs, [])

    def test_annotation_wir_init(self):
        mock_wir = MagicMock(spec=nx.DiGraph)
        mock_wir.copy.return_value = MagicMock(spec=nx.DiGraph)
        mock_prs = MagicMock()
        mock_kb = MagicMock()
        
        annotator = AnnotationWIR(mock_wir, mock_prs, mock_kb)
        self.assertEqual(annotator.wir, mock_wir)
        self.assertEqual(annotator.prs, mock_prs)
        self.assertEqual(annotator.knowledge_base, mock_kb)
        self.assertEqual(annotator.annotated_wir, mock_wir.copy.return_value)

    def test_get_annotation(self):
        annotator = AnnotationWIR(nx.DiGraph(), [], MagicMock())
        annotator.annotated_wir.add_node('node1', annotations=['anno1'])
        self.assertEqual(annotator._get_annotation('node1'), ['anno1'])
        self.assertIsNone(annotator._get_annotation('node2'))

    @patch.object(AnnotationWIR, 'find_import_nodes', return_value=['import_node'])
    @patch.object(AnnotationWIR, 'extract_library_and_module', return_value=('lib', 'mod'))
    @patch.object(AnnotationWIR, '_extract_pr_elements', return_value=(['in'], 'caller', 'process', ['out']))
    @patch.object(AnnotationWIR, 'check_if_visited', return_value=False)
    @patch.object(AnnotationWIR, 'visit_node')
    @patch.object(AnnotationWIR, '_get_annotation', side_effect=[None, None]) # For caller and previous_input
    @patch.object(AnnotationWIR, '_annotate_node', side_effect=lambda n, a: (n, a))
    @patch.object(AnnotationWIR, 'find_forward_prs', return_value=[])
    def test_annotate_basic_flow(self, mock_find_forward_prs, mock_annotate_node, mock_get_annotation, mock_visit_node, mock_check_if_visited, mock_extract_pr_elements, mock_extract_lib_mod, mock_find_import_nodes):
        mock_wir = nx.DiGraph()
        mock_wir.add_node('import_node')
        mock_wir.add_node('process')
        mock_wir.add_node('out')
        mock_wir.add_node('in')
        mock_wir.add_node('caller')

        mock_kb = MagicMock()
        mock_kb.return_value = (['input_anno'], ['output_anno'])
        mock_kb.back_query.return_value = ['back_input_anno']

        annotator = AnnotationWIR(mock_wir, [], mock_kb)
        annotator.annotate()

        mock_find_import_nodes.assert_called_once()
        mock_extract_lib_mod.assert_called_once_with('import_node')
        mock_extract_pr_elements.assert_called_once_with('import_node')
        mock_check_if_visited.assert_called_once_with('process')
        mock_visit_node.assert_called_once_with('process')
        mock_kb.assert_called_once_with('lib', 'mod', None, 'process')
        mock_annotate_node.assert_has_calls([
            call('out', 'output_anno'),
            call('in', 'input_anno')
        ])
        mock_kb.back_query.assert_called_once()
        mock_find_forward_prs.assert_called_once()

    def test_find_import_nodes(self):
        wir = nx.DiGraph()
        wir.add_node('importas_pandas')
        wir.add_node('some_other_node')
        wir.add_node('importfrom_sklearn')
        annotator = AnnotationWIR(wir, [], MagicMock())
        self.assertEqual(set(annotator.find_import_nodes()), {'importas_pandas', 'importfrom_sklearn'})

    def test_extract_library_and_module(self):
        wir = nx.DiGraph()
        wir.add_node('pandas', label='pandas')
        wir.add_node('read_csv')
        wir.add_edge('pandas', 'read_csv', edge_type='input_to_operation')
        annotator = AnnotationWIR(wir, [], MagicMock())
        lib, mod = annotator.extract_library_and_module('read_csv')
        self.assertEqual(lib, 'pandas')
        self.assertIsNone(mod)

        wir_sub = nx.DiGraph()
        wir_sub.add_node('sklearn.model_selection', label='sklearn.model_selection')
        wir_sub.add_node('train_test_split')
        wir_sub.add_edge('sklearn.model_selection', 'train_test_split', edge_type='input_to_operation')
        annotator_sub = AnnotationWIR(wir_sub, [], MagicMock())
        lib_sub, mod_sub = annotator_sub.extract_library_and_module('train_test_split')
        self.assertEqual(lib_sub, 'sklearn')
        self.assertEqual(mod_sub, 'model_selection')

    def test_extract_pr_elements_operation(self):
        wir = nx.DiGraph()
        wir.add_node('input1')
        wir.add_node('caller1')
        wir.add_node('operation1')
        wir.add_node('output1')
        wir.add_edge('input1', 'operation1', edge_type='input_to_operation')
        wir.add_edge('caller1', 'operation1', edge_type='caller_to_operation')
        wir.add_edge('operation1', 'output1', edge_type='operation_to_output')
        annotator = AnnotationWIR(wir, [], MagicMock())
        inputs, caller, process, outputs = annotator._extract_pr_elements('operation1', node_type='operation')
        self.assertEqual(inputs, ['input1'])
        self.assertEqual(caller, 'caller1')
        self.assertEqual(process, 'operation1')
        self.assertEqual(outputs, ['output1'])

    def test_extract_pr_elements_output(self):
        wir = nx.DiGraph()
        wir.add_node('operation1')
        wir.add_node('output1')
        wir.add_edge('operation1', 'output1', edge_type='operation_to_output')
        annotator = AnnotationWIR(wir, [], MagicMock())
        inputs, caller, process, outputs = annotator._extract_pr_elements('output1', node_type='output')
        self.assertEqual(inputs, ['input1'])
        self.assertEqual(caller, 'caller1')
        self.assertEqual(process, 'operation1')
        self.assertEqual(outputs, ['output1'])

    def test_annotate_node(self):
        wir = nx.DiGraph()
        wir.add_node('node1')
        annotator = AnnotationWIR(wir, [], MagicMock())
        annotator._annotate_node('node1', 'annotation_value')
        self.assertEqual(annotator.annotated_wir.nodes['node1']['annotations'], ['annotation_value'])

    def test_check_if_visited(self):
        wir = nx.DiGraph()
        wir.add_node('node1')
        annotator = AnnotationWIR(wir, [], MagicMock())
        self.assertFalse(annotator.check_if_visited('node1'))
        annotator.visit_node('node1')
        self.assertTrue(annotator.check_if_visited('node1'))
        self.assertTrue(annotator.check_if_visited(None))

    def test_visit_node(self):
        wir = nx.DiGraph()
        wir.add_node('node1')
        annotator = AnnotationWIR(wir, [], MagicMock())
        annotator.visit_node('node1')
        self.assertTrue(annotator.annotated_wir.nodes['node1']['visited'])

    def test_find_forward_prs(self):
        prs = [
            (['in1'], 'c1', 'op1', ['out1']),
            (['out1'], 'c2', 'op2', ['out2']),
            (['in2'], 'c3', 'op3', ['out3'])
        ]
        annotator = AnnotationWIR(nx.DiGraph(), prs, MagicMock())
        forward_prs = annotator.find_forward_prs((['in0'], 'c0', 'op0', ['out1']))
        self.assertEqual(set(forward_prs), {'op2'})

    @patch('matplotlib.pyplot.figure')
    @patch('networkx.drawing.nx_agraph.graphviz_layout', return_value={})
    @patch('networkx.draw_networkx_nodes')
    @patch('networkx.draw_networkx_edges')
    @patch('networkx.draw_networkx_labels')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_draw_graph(self, mock_close, mock_savefig, mock_legend, mock_labels, mock_edges, mock_nodes, mock_layout, mock_figure):
        annotator = AnnotationWIR(nx.DiGraph(), [], MagicMock())
        annotator.draw_graph([], [], [], [], "test.png")
        mock_figure.assert_called_once()
        mock_layout.assert_called_once()
        mock_nodes.assert_called_once()
        mock_edges.assert_called_once()
        mock_labels.assert_called_once()
        mock_legend.assert_called_once()
        mock_savefig.assert_called_once_with("test.png")
        mock_close.assert_called_once()

if __name__ == '__main__':
    unittest.main()
