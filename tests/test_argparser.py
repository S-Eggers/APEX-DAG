import unittest
from ApexDAG.argparser import argparser
import argparse


class TestArgparser(unittest.TestCase):
    def test_argparser_instance(self):
        self.assertIsInstance(argparser, argparse.ArgumentParser)

    def test_default_values(self):
        args = argparser.parse_args([])
        self.assertEqual(args.notebook, "./data/raw/demo.ipynb")
        self.assertEqual(args.window, 1)
        self.assertFalse(args.greedy)
        self.assertFalse(args.save_prev)
        self.assertEqual(args.experiment, "data_flow_graph_test")
        self.assertFalse(args.draw)
        self.assertIsNone(args.checkpoint_path)
        self.assertEqual(
            args.config_path, "ApexDAG/experiments/configs/pretrain/default.yaml"
        )

    def test_argument_parsing(self):
        args = argparser.parse_args(
            [
                "--notebook",
                "test.ipynb",
                "--window",
                "10",
                "--greedy",
                "--save_prev",
                "--experiment",
                "my_experiment",
                "--draw",
                "--checkpoint_path",
                "/path/to/checkpoint",
                "--config_path",
                "/path/to/config.yaml",
            ]
        )
        self.assertEqual(args.notebook, "test.ipynb")
        self.assertEqual(args.window, 10)
        self.assertTrue(args.greedy)
        self.assertTrue(args.save_prev)
        self.assertEqual(args.experiment, "my_experiment")
        self.assertTrue(args.draw)
        self.assertEqual(args.checkpoint_path, "/path/to/checkpoint")
        self.assertEqual(args.config_path, "/path/to/config.yaml")


if __name__ == "__main__":
    unittest.main()
