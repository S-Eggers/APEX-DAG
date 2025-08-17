import unittest
from unittest.mock import patch
from ApexDAG.util.training_utils import (
    DOMAIN_LABEL_TO_SUBSAMPLE,
    TASKS_PER_GRAPH_TRANSFORM_MODE_FINETUNE,
    TASKS_PER_GRAPH_TRANSFORM_MODE_PRETRAIN,
    GraphTransformsMode,
    InsufficientNegativeEdgesException,
    InsufficientPositiveEdgesException,
    set_seed,
)


class TestTrainingUtils(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(DOMAIN_LABEL_TO_SUBSAMPLE, "DATA_TRANSFORM")
        self.assertIsInstance(TASKS_PER_GRAPH_TRANSFORM_MODE_FINETUNE, dict)
        self.assertIsInstance(TASKS_PER_GRAPH_TRANSFORM_MODE_PRETRAIN, dict)
        self.assertEqual(
            TASKS_PER_GRAPH_TRANSFORM_MODE_FINETUNE["REVERSED"], ["node_classification"]
        )
        self.assertEqual(
            TASKS_PER_GRAPH_TRANSFORM_MODE_PRETRAIN["ORIGINAL"],
            ["node_classification", "edge_classification", "edge_existence"],
        )

    def test_graph_transforms_mode_enum(self):
        self.assertEqual(GraphTransformsMode.REVERSED.value, "REVERSED")
        self.assertEqual(GraphTransformsMode.ORIGINAL.value, "ORIGINAL")
        self.assertEqual(GraphTransformsMode.REVERSED_MASKED.value, "REVERSED_MASKED")
        self.assertEqual(GraphTransformsMode.ORIGINAL_MASKED.value, "ORIGINAL_MASKED")

    def test_insufficient_negative_edges_exception(self):
        with self.assertRaisesRegex(
            InsufficientNegativeEdgesException, "Test message"
        ):  # Use Regex for message matching
            raise InsufficientNegativeEdgesException("Test message")
        self.assertTrue(issubclass(InsufficientNegativeEdgesException, Exception))

    def test_insufficient_positive_edges_exception(self):
        with self.assertRaisesRegex(
            InsufficientPositiveEdgesException, "Another message"
        ):  # Use Regex for message matching
            raise InsufficientPositiveEdgesException("Another message")
        self.assertTrue(issubclass(InsufficientPositiveEdgesException, Exception))

    @patch("os.environ.__setitem__")
    @patch("random.seed")
    @patch("numpy.random.seed")
    @patch("torch.manual_seed")
    @patch("torch.cuda.manual_seed_all")
    @patch("torch.backends.cudnn.deterministic")
    @patch("torch.backends.cudnn.benchmark")
    def test_set_seed(
        self,
        mock_cudnn_benchmark,
        mock_cudnn_deterministic,
        mock_cuda_manual_seed_all,
        mock_manual_seed,
        mock_np_random_seed,
        mock_random_seed,
        mock_os_environ_setitem,
    ):
        seed_value = 42
        set_seed(seed_value)

        mock_os_environ_setitem.assert_called_once_with(
            "PYTHONHASHSEED", str(seed_value)
        )
        mock_random_seed.assert_called_once_with(seed_value)
        mock_np_random_seed.assert_called_once_with(seed_value)
        mock_manual_seed.assert_called_once_with(seed_value)
        mock_cuda_manual_seed_all.assert_called_once_with(seed_value)
        self.assertTrue(mock_cudnn_deterministic)
        self.assertFalse(mock_cudnn_benchmark)


if __name__ == "__main__":
    unittest.main()
