import unittest
from unittest.mock import MagicMock, patch, call
import ast
import logging
from ApexDAG.vamsa.utils import (
    WIRNode,
    check_bipartie,
    add_id,
    remove_id,
    is_empty_or_none_list,
    flatten,
    get_relevant_code,
    remove_comment_lines,
    merge_prs
)

class TestVamsaUtils(unittest.TestCase):

    def test_wir_node_dataclass(self):
        node = WIRNode("test_node")
        self.assertEqual(node.node, "test_node")
        self.assertFalse(node.isAttribute)

        node_attr = WIRNode("test_node", isAttribute=True)
        self.assertTrue(node_attr.isAttribute)

    @patch('ApexDAG.vamsa.utils.logger.warning')
    def test_check_bipartie_true(self, mock_warning):
        prs = {("a", "b", "op1", "c"), ("x", "y", "op2", "z")}
        self.assertTrue(check_bipartie(prs))
        mock_warning.assert_not_called()

    @patch('ApexDAG.vamsa.utils.logger.warning')
    def test_check_bipartie_false(self, mock_warning):
        prs = {("a", "b", "op1", "c"), ("op1", "y", "op2", "z")}
        self.assertFalse(check_bipartie(prs))
        mock_warning.assert_called_once()

    @patch('random.randint', return_value=1234)
    def test_add_id(self, mock_randint):
        self.assertEqual(add_id(), ":id1234")
        mock_randint.assert_called_once_with(0, 2500)

    def test_remove_id(self):
        self.assertEqual(remove_id("node_name:id123"), "node_name")
        self.assertEqual(remove_id("node_name"), "node_name")
        self.assertEqual(remove_id(None), "")

    def test_is_empty_or_none_list(self):
        self.assertTrue(is_empty_or_none_list(None))
        self.assertTrue(is_empty_or_none_list([]))
        self.assertTrue(is_empty_or_none_list([None, None]))
        self.assertFalse(is_empty_or_none_list([1, 2]))
        self.assertFalse(is_empty_or_none_list([None, 1]))
        self.assertFalse(is_empty_or_none_list("not a list"))

    def test_flatten(self):
        self.assertEqual(list(flatten([1, [2, 3], 4])), [1, 2, 3, 4])
        self.assertEqual(list(flatten([1, [2, [3, 4]], 5])), [1, 2, 3, 4, 5])
        self.assertEqual(list(flatten([])), [])
        self.assertEqual(list(flatten(["a", ["b", "c"]])), ["a", "b", "c"])

    def test_get_relevant_code(self):
        mock_node = MagicMock()
        mock_node.lineno = 2
        mock_node.col_offset = 4
        mock_node.end_col_offset = 9
        mock_node.__dict__ = {'lineno': 2, 'col_offset': 4, 'end_col_offset': 9}
        file_lines = ["line1", "  test_code", "line3"]
        self.assertEqual(get_relevant_code(mock_node, file_lines), "_code")

        mock_node_no_attr = MagicMock()
        mock_node_no_attr.__dict__ = {}
        self.assertIsNone(get_relevant_code(mock_node_no_attr, file_lines))

    def test_remove_comment_lines(self):
        code = """
# This is a comment
print("hello")
x = 1 # inline comment
# Another comment
"""
        expected_code = """
x = 1 # inline comment
"""
        self.assertEqual(remove_comment_lines(code), expected_code)

        self.assertEqual(remove_comment_lines(""), "")
        self.assertEqual(remove_comment_lines("#only comment"), "")
        self.assertEqual(remove_comment_lines("print('test')"), "")

    def test_merge_prs(self):
        p1 = [("a", "b", "op1", "c"), ("x", "y", "op2", "z")]
        p2 = [("m", "n", "op3", "o"), ("a", "b", "op1", "c")] # Duplicate
        merged = merge_prs(p1, p2)
        self.assertEqual(len(merged), 3)
        self.assertIn(("a", "b", "op1", "c"), merged)
        self.assertIn(("x", "y", "op2", "z"), merged)
        self.assertIn(("m", "n", "op3", "o"), merged)

if __name__ == '__main__':
    unittest.main()
