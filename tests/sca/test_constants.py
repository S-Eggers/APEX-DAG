import unittest
from ApexDAG.sca.constants import (
    NODE_TYPES,
    EDGE_TYPES,
    DOMAIN_EDGE_TYPES,
    VERBOSE,
    REVERSE_NODE_TYPES,
    REVERSE_EDGE_TYPES,
    REVERSE_DOMAIN_EDGE_TYPES,
)


class TestConstants(unittest.TestCase):
    def test_node_types(self):
        self.assertIsInstance(NODE_TYPES, dict)
        self.assertIn("VARIABLE", NODE_TYPES)
        self.assertEqual(NODE_TYPES["VARIABLE"], 0)
        # Add more assertions for other expected keys and values
        self.assertEqual(len(NODE_TYPES), 8)

    def test_edge_types(self):
        self.assertIsInstance(EDGE_TYPES, dict)
        self.assertIn("CALLER", EDGE_TYPES)
        self.assertEqual(EDGE_TYPES["CALLER"], 0)
        # Add more assertions for other expected keys and values
        self.assertEqual(len(EDGE_TYPES), 6)

    def test_domain_edge_types(self):
        self.assertIsInstance(DOMAIN_EDGE_TYPES, dict)
        self.assertIn("MODEL_TRAIN", DOMAIN_EDGE_TYPES)
        self.assertEqual(DOMAIN_EDGE_TYPES["MODEL_TRAIN"], 0)
        # Add more assertions for other expected keys and values
        self.assertEqual(len(DOMAIN_EDGE_TYPES), 9)

    def test_verbose(self):
        self.assertIsInstance(VERBOSE, bool)
        self.assertFalse(VERBOSE)

    def test_reverse_node_types(self):
        self.assertIsInstance(REVERSE_NODE_TYPES, dict)
        self.assertEqual(len(REVERSE_NODE_TYPES), len(NODE_TYPES))
        for key, value in NODE_TYPES.items():
            self.assertEqual(REVERSE_NODE_TYPES[value], key)

    def test_reverse_edge_types(self):
        self.assertIsInstance(REVERSE_EDGE_TYPES, dict)
        self.assertEqual(len(REVERSE_EDGE_TYPES), len(EDGE_TYPES))
        for key, value in EDGE_TYPES.items():
            self.assertEqual(REVERSE_EDGE_TYPES[value], key)

    def test_reverse_domain_edge_types(self):
        self.assertIsInstance(REVERSE_DOMAIN_EDGE_TYPES, dict)
        self.assertEqual(len(REVERSE_DOMAIN_EDGE_TYPES), 5)
        self.assertEqual(REVERSE_DOMAIN_EDGE_TYPES[0], "MODEL_TRAIN_EVALUATION")
        self.assertEqual(REVERSE_DOMAIN_EDGE_TYPES[1], "DATA_IMPORT_EXTRACTION")
        self.assertEqual(REVERSE_DOMAIN_EDGE_TYPES[2], "DATA_TRANSFORM")
        self.assertEqual(REVERSE_DOMAIN_EDGE_TYPES[3], "EDA")
        self.assertEqual(REVERSE_DOMAIN_EDGE_TYPES[4], "ENVIRONMENT+DATA_EXPORT")


if __name__ == "__main__":
    unittest.main()
