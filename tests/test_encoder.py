import unittest
from unittest.mock import MagicMock, patch
import networkx as nx
import torch
from torch_geometric.data import Data
from ApexDAG.encoder import Encoder
from ApexDAG.util.training_utils import (
    InsufficientNegativeEdgesException,
    InsufficientPositiveEdgesException,
)


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock()
        self.encoder = Encoder(logger=self.mock_logger)

    def _create_sample_graph(self):
        graph = nx.MultiDiGraph()
        graph.add_node(0, label="a_var", node_type=1)
        graph.add_node(1, label="b_var", node_type=2)
        graph.add_node(2, label="c_var", node_type=3)
        graph.add_edge(0, 1, code="a = b", edge_type="assignment")
        graph.add_edge(1, 2, code="b = c", edge_type="assignment")
        return graph

    def test_init(self):
        self.assertIsNotNone(self.encoder._embedding)
        self.assertEqual(self.encoder.nude_num_types_pretrain, 8)
        self.assertEqual(self.encoder.edge_num_types_pretrain, 6)
        self.assertEqual(self.encoder.min_edges, 2)

    @patch("ApexDAG.encoder.Embedding")
    def test_encode(self, mock_embedding):
        mock_embedding.return_value.embed.return_value = torch.zeros(300)
        graph = self._create_sample_graph()
        data = self.encoder.encode(graph, "edge_type")

        self.assertIsInstance(data, Data)
        self.assertTrue("x" in data)
        self.assertTrue("node_types" in data)
        self.assertTrue("edge_index" in data)
        self.assertTrue("edge_types" in data)
        self.assertTrue("edge_features" in data)
        self.assertTrue("edge_existence" in data)

    @patch("ApexDAG.encoder.Embedding")
    def test_encode_reversed(self, mock_embedding):
        mock_embedding.return_value.embed.return_value = torch.zeros(300)
        graph = self._create_sample_graph()
        data = self.encoder.encode_reversed(graph, "edge_type")

        self.assertIsInstance(data, Data)
        self.assertTrue("x" in data)
        self.assertTrue("node_types" in data)
        self.assertTrue("edge_index" in data)
        self.assertTrue("edge_types" in data)
        self.assertTrue("edge_features" in data)

    def test_sample_negative_edges(self):
        graph = self._create_sample_graph()
        node_to_id = {node: idx for idx, node in enumerate(graph.nodes())}
        negative_edges = self.encoder._sample_negative_edges(graph, node_to_id, 1)
        self.assertEqual(len(negative_edges), 1)

    def test_insufficient_negative_edges(self):
        graph = nx.complete_graph(2, create_using=nx.MultiDiGraph)
        node_to_id = {node: idx for idx, node in enumerate(graph.nodes())}
        with self.assertRaises(InsufficientNegativeEdgesException):
            self.encoder._sample_negative_edges(graph, node_to_id, 1)

    @patch("ApexDAG.encoder.Embedding")
    def test_extract_node_features(self, mock_embedding):
        mock_embedding.return_value.embed.return_value = torch.zeros(300)
        graph = self._create_sample_graph()
        node_features, node_types = self.encoder._extract_node_features(graph)

        self.assertIsInstance(node_features, torch.Tensor)
        self.assertEqual(node_features.shape, (3, 300))
        self.assertEqual(node_types, [1, 2, 3])

    @patch("ApexDAG.encoder.Embedding")
    def test_extract_edge_features(self, mock_embedding):
        mock_embedding.return_value.embed.return_value = torch.zeros(300)
        graph = self._create_sample_graph()
        edge_features, edge_types, edge_existence = self.encoder._extract_edge_features(
            graph, [], "edge_type"
        )

        self.assertIsInstance(edge_features, torch.Tensor)
        self.assertEqual(edge_features.shape, (2, 300))
        self.assertIsInstance(edge_types, torch.Tensor)
        self.assertIsInstance(edge_existence, torch.Tensor)

    def test_insufficient_positive_edges(self):
        graph = nx.MultiDiGraph()
        with self.assertRaises(InsufficientPositiveEdgesException):
            self.encoder._extract_edge_features(graph, [], "edge_type")


if __name__ == "__main__":
    unittest.main()
