import unittest
from unittest.mock import MagicMock, patch, call
import networkx as nx
import os
import json
import matplotlib.pyplot as plt
from ApexDAG.util.draw import Draw
from ApexDAG.sca.constants import NODE_TYPES, EDGE_TYPES

class TestDraw(unittest.TestCase):

    def setUp(self):
        self.draw_instance = Draw(NODE_TYPES, EDGE_TYPES)
        self.mock_graph = nx.DiGraph()
        self.mock_graph.add_node(0, label="node0", node_type=NODE_TYPES["VARIABLE"])
        self.mock_graph.add_node(1, label="node1", node_type=NODE_TYPES["FUNCTION"])
        self.mock_graph.add_edge(0, 1, code="edge_code", edge_type=EDGE_TYPES["CALLER"], count=1, predicted_label="")

    @patch('os.path.exists', return_value=False)
    @patch('os.makedirs')
    @patch('os.path.join', side_effect=lambda *args: '/'.join(args))
    @patch('os.getcwd', return_value='/mock_cwd')
    @patch.object(Draw, 'dfg_to_json', return_value='{"elements":[]}')
    @patch('builtins.open', MagicMock())
    def test_dfg_webrendering(self, mock_open, mock_dfg_to_json, mock_getcwd, mock_join, mock_makedirs, mock_exists):
        self.draw_instance.dfg_webrendering(self.mock_graph, save_path="/path/to/test_graph.json")
        mock_exists.assert_called_once_with("/mock_cwd/path/to")
        mock_makedirs.assert_called_once_with("/mock_cwd/path/to", exist_ok=True)
        mock_dfg_to_json.assert_called_once_with(self.mock_graph)
        mock_open.assert_called_once_with("/mock_cwd/path/to/test_graph.json", "w")
        mock_open.return_value.write.assert_called_once_with('{"elements":[]}')

    def test_dfg_to_json(self):
        json_output = self.draw_instance.dfg_to_json(self.mock_graph)
        expected_output = {
            "elements": [
                {"data": {"id": "0", "label": "node0", "node_type": NODE_TYPES["VARIABLE"]}},
                {"data": {"id": "1", "label": "node1", "node_type": NODE_TYPES["FUNCTION"]}},
                {"data": {"source": "0", "target": "1", "edge_type": EDGE_TYPES["CALLER"], "label": "edge_code", "predicted_label": ""}}
            ]
        }
        self.assertEqual(json.loads(json_output), expected_output)

    @patch('os.path.exists', return_value=False)
    @patch('os.makedirs')
    @patch('os.path.join', side_effect=lambda *args: '/'.join(args))
    @patch('os.getcwd', return_value='/mock_cwd')
    @patch('matplotlib.pyplot.figure')
    @patch('networkx.nx_agraph.write_dot')
    @patch('networkx.drawing.nx_agraph.graphviz_layout', return_value={0: (0,0), 1: (1,1)})
    @patch('networkx.draw_networkx_edges')
    @patch('networkx.draw_networkx_nodes')
    @patch('networkx.draw_networkx_labels')
    @patch('networkx.draw_networkx_edge_labels')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.close')
    def test_dfg(self, mock_close, mock_clf, mock_savefig, mock_axis, mock_legend, mock_edge_labels, mock_labels, mock_nodes, mock_edges, mock_layout, mock_write_dot, mock_figure, mock_getcwd, mock_join, mock_makedirs, mock_exists):
        self.draw_instance.dfg(self.mock_graph, save_path="/path/to/test_dfg.png")
        mock_exists.assert_called_once_with("/mock_cwd/path/to")
        mock_makedirs.assert_called_once_with("/mock_cwd/path/to", exist_ok=True)
        mock_figure.assert_called_once_with(figsize=(32, 64))
        mock_write_dot.assert_called_once_with(self.mock_graph, "/mock_cwd/path/to/test_dfg.dot")
        mock_layout.assert_called_once_with(self.mock_graph, prog="dot")
        mock_edges.assert_called_once()
        mock_nodes.assert_called_once()
        mock_labels.assert_called_once()
        mock_edge_labels.assert_called_once()
        mock_legend.assert_called_once()
        mock_axis.assert_called_once_with('off')
        mock_savefig.assert_called_once_with("/mock_cwd/path/to/test_dfg.png")
        mock_clf.assert_called_once()
        mock_close.assert_called_once()

    @patch('os.path.exists', return_value=False)
    @patch('os.makedirs')
    @patch('os.path.join', side_effect=lambda *args: '/'.join(args))
    @patch('os.getcwd', return_value='/mock_cwd')
    @patch('matplotlib.pyplot.figure')
    @patch('networkx.nx_agraph.write_dot')
    @patch('networkx.drawing.nx_agraph.graphviz_layout', return_value={0: (0,0), 1: (1,1)})
    @patch('networkx.draw_networkx_edges')
    @patch('networkx.draw_networkx_nodes')
    @patch('networkx.draw_networkx_labels')
    @patch('networkx.draw_networkx_edge_labels')
    @patch('matplotlib.pyplot.legend')
    @patch('matplotlib.pyplot.axis')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.clf')
    @patch('matplotlib.pyplot.close')
    def test_labelled_dfg(self, mock_close, mock_clf, mock_savefig, mock_axis, mock_legend, mock_edge_labels, mock_labels, mock_nodes, mock_edges, mock_layout, mock_write_dot, mock_figure, mock_getcwd, mock_join, mock_makedirs, mock_exists):
        self.mock_graph.edges[0,1]['domain_label'] = 'MODEL_TRAIN'
        self.draw_instance.labelled_dfg(self.mock_graph, save_path="/path/to/test_labelled_dfg.png")
        mock_exists.assert_called_once_with("/mock_cwd/path/to")
        mock_makedirs.assert_called_once_with("/mock_cwd/path/to", exist_ok=True)
        mock_figure.assert_called_once_with(figsize=(32, 64))
        mock_write_dot.assert_called_once_with(self.mock_graph, "/mock_cwd/path/to/test_labelled_dfg.dot")
        mock_layout.assert_called_once_with(self.mock_graph, prog="dot")
        mock_edges.assert_called_once()
        mock_nodes.assert_called_once()
        mock_labels.assert_called_once()
        mock_edge_labels.assert_called_once()
        mock_legend.assert_called_once()
        mock_axis.assert_called_once_with('off')
        mock_savefig.assert_called_once_with("/mock_cwd/path/to/test_labelled_dfg.png")
        mock_clf.assert_called_once()
        mock_close.assert_called_once()

    @patch('os.path.exists', return_value=False)
    @patch('os.makedirs')
    @patch('os.path.join', side_effect=lambda *args: '/'.join(args))
    @patch('os.getcwd', return_value='/mock_cwd')
    @patch('matplotlib.pyplot.figure')
    @patch('networkx.nx_agraph.write_dot')
    @patch('networkx.drawing.nx_agraph.graphviz_layout', return_value={0: (0,0), 1: (1,1)})
    @patch('networkx.draw')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.clf')
    def test_ast(self, mock_clf, mock_savefig, mock_draw, mock_layout, mock_write_dot, mock_figure, mock_getcwd, mock_join, mock_makedirs, mock_exists):
        self.draw_instance.ast(self.mock_graph, t2t_paths=[[0, 1]])
        mock_exists.assert_called_once_with("/mock_cwd/output")
        mock_makedirs.assert_called_once_with("/mock_cwd/output", exist_ok=True)
        mock_figure.assert_called_once_with(figsize=(128, 32))
        mock_write_dot.assert_called_once_with(self.mock_graph, "/mock_cwd/output/ast.dot")
        mock_layout.assert_called_once_with(self.mock_graph, prog="dot")
        mock_draw.assert_called_once()
        mock_savefig.assert_called_once_with("/mock_cwd/output/ast_graph.png")
        mock_clf.assert_called_once()

if __name__ == '__main__':
    unittest.main()
