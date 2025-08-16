import unittest
from unittest.mock import MagicMock, patch, mock_open, call
import networkx as nx
import torch
from ApexDAG.util.networkx_to_pyc import write_dict_to_file, load_dict_from_file, split_code, networkx_to_pyc, node2vec_embedding

class TestNetworkxToPyc(unittest.TestCase):

    def test_write_dict_to_file(self):
        mock_file = mock_open()
        with patch('builtins.open', mock_file):
            write_dict_to_file("test.txt", {"a": "1", "b": "2"})
            mock_file.assert_called_once_with("test.txt", "w")
            mock_file().write.assert_has_calls([
                call("a: 1\n"),
                call("b: 2\n")
            ])

    def test_load_dict_from_file(self):
        mock_file = mock_open(read_data="a: 1\nb: 2\n")
        with patch('builtins.open', mock_file):
            dictionary = load_dict_from_file("test.txt")
            self.assertEqual(dictionary, {"a": "1", "b": "2"})
            mock_file.assert_called_once_with("test.txt", "r")

    def test_split_code(self):
        self.assertEqual(split_code("hello world"), ["hello", "world"])
        self.assertEqual(split_code("single"), ["single"])
        self.assertEqual(split_code(""), [""])

    @patch('torch.nn.Embedding')
    @patch('torch_geometric.utils.from_networkx')
    @patch('ApexDAG.util.networkx_to_pyc.write_dict_to_file')
    @patch('torch.tensor', side_effect=lambda x: x) # Mock torch.tensor to return input directly
    def test_networkx_to_pyc(self, mock_torch_tensor, mock_write_dict_to_file, mock_from_networkx, mock_embedding):
        g = nx.Graph()
        g.add_node(0, label="node0_label", code="node0_code")
        g.add_node(1, label="node1_label", code="node1_code")

        mock_embedding_instance = MagicMock()
        mock_embedding.return_value = mock_embedding_instance
        mock_embedding_instance.side_effect = lambda x: f"embedding_{x}"

        networkx_to_pyc(g)

        mock_embedding.assert_called_once_with(4, 256) # 4 unique words: node0_label, node0_code, node1_label, node1_code
        mock_write_dict_to_file.assert_called_once_with("output/word2idx.txt", {
            'node0_label': 0,
            'node0_code': 1,
            'node1_label': 2,
            'node1_code': 3
        })
        mock_from_networkx.assert_called_once()
        # Verify that attributes other than 'embedding' are deleted
        self.assertNotIn('label', g.nodes[0])
        self.assertNotIn('code', g.nodes[0])
        self.assertIn('embedding', g.nodes[0])

    @patch('torch.cuda.is_available', return_value=False)
    @patch('torch_geometric.nn.models.Node2Vec')
    def test_node2vec_embedding(self, mock_node2vec, mock_is_available):
        mock_data = MagicMock()
        mock_model_instance = MagicMock()
        mock_node2vec.return_value = mock_model_instance
        mock_model_instance.loader.return_value = []
        mock_model_instance.test.return_value = 0.5
        mock_model_instance.return_value = MagicMock()

        embedding = node2vec_embedding(mock_data)

        mock_node2vec.assert_called_once_with(
            mock_data.edge_index, embedding_dim=256, walk_length=20,
            context_size=10, walks_per_node=10,
            num_negative_samples=1, p=1, q=1, sparse=True
        )
        mock_model_instance.loader.assert_called_once_with(batch_size=128, shuffle=True, num_workers=4)
        mock_model_instance.train.assert_called_once()
        mock_model_instance.eval.assert_called_once()
        mock_model_instance.test.assert_called_once()
        self.assertIsNotNone(embedding)

if __name__ == '__main__':
    unittest.main()
