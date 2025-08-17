import unittest
from unittest.mock import MagicMock, patch
import networkx as nx
from ApexDAG.sca.graph_utils import (
    convert_multidigraph_to_digraph,
    get_subgraph,
    get_all_subgraphs,
    debug_graph,
    save_graph,
    load_graph,
)
from ApexDAG.sca.constants import NODE_TYPES, EDGE_TYPES


class TestGraphUtils(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock()
        # Patch setup_logging to return our mock logger
        patcher = patch(
            "ApexDAG.util.logging.setup_logging", return_value=self.mock_logger
        )
        self.mock_setup_logging = patcher.start()
        self.addCleanup(patcher.stop)

    def test_convert_multidigraph_to_digraph_no_multiple_edges(self):
        G = nx.MultiDiGraph()
        G.add_node("a", label="node_a", node_type=NODE_TYPES["VARIABLE"])
        G.add_node("b", label="node_b", node_type=NODE_TYPES["VARIABLE"])
        G.add_edge("a", "b", code="edge_code", edge_type=EDGE_TYPES["CALLER"])

        new_G = convert_multidigraph_to_digraph(G, NODE_TYPES)

        self.assertIsInstance(new_G, nx.DiGraph)
        self.assertEqual(new_G.number_of_nodes(), 2)
        self.assertEqual(new_G.number_of_edges(), 1)
        self.assertTrue(new_G.has_edge("a", "b"))
        self.assertEqual(new_G["a"]["b"]["code"], "edge_code")

    def test_convert_multidigraph_to_digraph_with_multiple_edges(self):
        G = nx.MultiDiGraph()
        G.add_node("a", label="node_a", node_type=NODE_TYPES["VARIABLE"])
        G.add_node("b", label="node_b", node_type=NODE_TYPES["VARIABLE"])
        G.add_edge("a", "b", key=0, code="edge_code_1", edge_type=EDGE_TYPES["CALLER"])
        G.add_edge("a", "b", key=1, code="edge_code_2", edge_type=EDGE_TYPES["INPUT"])

        new_G = convert_multidigraph_to_digraph(G, NODE_TYPES)

        self.assertIsInstance(new_G, nx.DiGraph)
        # Expecting original nodes + 1 intermediate node
        self.assertEqual(new_G.number_of_nodes(), 3)
        # Expecting 2 edges: a -> intermediate, intermediate -> b
        self.assertEqual(new_G.number_of_edges(), 2)
        self.assertTrue(new_G.has_node("b_intermediate_1"))
        self.assertTrue(new_G.has_edge("a", "b_intermediate_1"))
        self.assertTrue(new_G.has_edge("b_intermediate_1", "b"))

    def test_convert_multidigraph_to_digraph_missing_node_label(self):
        G = nx.MultiDiGraph()
        G.add_node("a", node_type=NODE_TYPES["VARIABLE"])
        with self.assertRaises(AttributeError) as cm:
            convert_multidigraph_to_digraph(G, NODE_TYPES)
        self.assertIn(
            "Node a is missing attribute(s): {'node_type': 0}", str(cm.exception)
        )

    def test_get_subgraph_exists(self):
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("d", "a")])
        variable_versions = {"a": ["a"]}
        subgraph = get_subgraph(G, variable_versions, "a")
        self.assertEqual(
            subgraph.nodes(), nx.DiGraph([("d", "a"), ("a", "b"), ("b", "c")]).nodes()
        )
        self.assertEqual(
            subgraph.edges(), nx.DiGraph([("d", "a"), ("a", "b"), ("b", "c")]).edges()
        )

    def test_get_subgraph_not_exists(self):
        G = nx.DiGraph()
        G.add_edges_from([("a", "b")])
        variable_versions = {"a": ["a"]}
        with self.assertRaises(ValueError):
            get_subgraph(G, variable_versions, "x")

    def test_get_all_subgraphs(self):
        G = nx.DiGraph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("d", "e")])
        variable_versions = {"a": ["a"], "d": ["d"]}
        subgraphs = get_all_subgraphs(G, variable_versions)
        self.assertEqual(len(subgraphs), 2)
        self.assertEqual(
            subgraphs[0].nodes(), nx.DiGraph([("a", "b"), ("b", "c")]).nodes()
        )
        self.assertEqual(subgraphs[1].nodes(), nx.DiGraph([("d", "e")]).nodes())

    @patch("ApexDAG.util.draw.Draw")
    @patch("logging.getLogger")
    def test_debug_graph_prev_exists(self, mock_get_logger, MockDraw):
        mock_get_logger.return_value = self.mock_logger
        # Mock os.path.exists to return True
        with patch("os.path.exists", return_value=True):
            # Create prev_G directly
            prev_G = nx.DiGraph()
            prev_G.add_node("a", label="node_a", node_type=0)
            prev_G.add_node("b", label="node_b", node_type=1)
            prev_G.add_edge("a", "b", code="old_code", edge_type=0)

            # Create G directly
            G = nx.DiGraph()
            G.add_node("a", label="node_a", node_type=0)
            G.add_node("b", label="node_b", node_type=1)
            G.add_node("c", label="node_c", node_type=0)  # Added node
            G.add_edge("a", "b", code="new_code", edge_type=0)  # Modified edge
            G.add_edge("b", "c", code="new_edge", edge_type=0)  # Added edge

            # Mock load_graph to return prev_G
            with patch("ApexDAG.sca.graph_utils.load_graph", return_value=prev_G):
                debug_graph(
                    G,
                    "prev.gml",
                    "new.gml",
                    NODE_TYPES,
                    EDGE_TYPES,
                    save_prev=True,
                    verbose=True,
                )

        # Assert that the logger was called with the expected debug messages
        debug_calls = [call[0][0] for call in self.mock_logger.debug.call_args_list]
        self.assertIn("Added nodes: ['c']", debug_calls)
        self.assertIn("Added edges: [('b', 'c')]", debug_calls)
        self.assertIn("Modified edges: [('a', 'b')]", debug_calls)

    @patch("ApexDAG.sca.graph_utils.load_graph")
    @patch("ApexDAG.sca.graph_utils.save_graph")
    @patch("os.path.exists")
    def test_debug_graph_prev_not_exists_save_prev_true(
        self, mock_os_path_exists, mock_save_graph, mock_load_graph
    ):
        mock_os_path_exists.return_value = False
        G = nx.DiGraph()
        debug_graph(G, "prev.gml", "new.gml", NODE_TYPES, EDGE_TYPES, save_prev=True)
        mock_save_graph.assert_called_once_with(G, "new.gml")
        mock_load_graph.assert_not_called()

    @patch("ApexDAG.sca.graph_utils.load_graph")
    @patch("ApexDAG.sca.graph_utils.save_graph")
    @patch("os.path.exists")
    def test_debug_graph_prev_not_exists_save_prev_false(
        self, mock_os_path_exists, mock_save_graph, mock_load_graph
    ):
        mock_os_path_exists.return_value = False
        G = nx.DiGraph()
        debug_graph(G, "prev.gml", "new.gml", NODE_TYPES, EDGE_TYPES, save_prev=False)
        mock_save_graph.assert_not_called()
        mock_load_graph.assert_not_called()

    @patch("networkx.write_gml")
    @patch("os.getcwd", return_value="/mock/cwd")
    def test_save_graph(self, mock_getcwd, mock_write_gml):
        G = nx.DiGraph()
        save_graph(G, "test.gml")
        mock_write_gml.assert_called_once_with(G, "/mock/cwd/test.gml")

    @patch("networkx.read_gml")
    @patch("os.path.exists", return_value=True)
    def test_load_graph_success(self, mock_os_path_exists, mock_read_gml):
        mock_graph = nx.DiGraph()
        mock_graph.add_node("a", label="node_a", node_type=0)
        mock_graph.add_node("b", label="node_b", node_type=0)
        mock_graph.add_edge("a", "b", code="edge_code", edge_type=0)
        mock_read_gml.return_value = mock_graph

        loaded_graph = load_graph("test.gml")
        self.assertEqual(loaded_graph, mock_graph)
        mock_os_path_exists.assert_called_once_with("test.gml")
        mock_read_gml.assert_called_once_with("test.gml")

    @patch("os.path.exists", return_value=False)
    def test_load_graph_file_not_found(self, mock_os_path_exists):
        with self.assertRaises(FileNotFoundError):
            load_graph("non_existent.gml")

    @patch("ApexDAG.sca.graph_utils.os.path.exists", return_value=True)
    @patch("networkx.read_gml")
    def test_load_graph_missing_node_type(self, mock_read_gml, mock_os_path_exists):
        mock_graph = nx.DiGraph()
        mock_graph.add_node("a", label="node_a")  # Missing node_type
        mock_read_gml.return_value = mock_graph

        with self.assertRaises(ValueError) as cm:
            load_graph("test.gml")
        self.assertIn(
            "Node a is missing required attribute 'node_type'", str(cm.exception)
        )

    @patch("ApexDAG.sca.graph_utils.os.path.exists", return_value=True)
    @patch("networkx.read_gml")
    def test_load_graph_missing_edge_attributes(
        self, mock_read_gml, mock_os_path_exists
    ):
        mock_graph = nx.DiGraph()
        mock_graph.add_node("a", label="node_a", node_type=0)
        mock_graph.add_node("b", label="node_b", node_type=0)
        mock_graph.add_edge("a", "b", code="edge_code")  # Missing edge_type
        mock_read_gml.return_value = mock_graph

        with self.assertRaises(ValueError) as cm:
            load_graph("test.gml")
        self.assertIn(
            "Edge a -> b is missing required attributes 'code' or 'edge_type",
            str(cm.exception),
        )


if __name__ == "__main__":
    unittest.main()
