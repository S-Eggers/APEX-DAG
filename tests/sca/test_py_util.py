import unittest
import ast
from ApexDAG.sca.py_util import get_operator_description, flatten_list

class TestPyUtil(unittest.TestCase):

    def test_get_operator_description_eq(self):
        node = ast.Compare(left=ast.Name(id='a', ctx=ast.Load()), ops=[ast.Eq()], comparators=[ast.Constant(value=1)])
        self.assertEqual(get_operator_description(node), "equal")

    def test_get_operator_description_not_eq(self):
        node = ast.Compare(left=ast.Name(id='a', ctx=ast.Load()), ops=[ast.NotEq()], comparators=[ast.Constant(value=1)])
        self.assertEqual(get_operator_description(node), "not equal")

    def test_get_operator_description_lt(self):
        node = ast.Compare(left=ast.Name(id='a', ctx=ast.Load()), ops=[ast.Lt()], comparators=[ast.Constant(value=1)])
        self.assertEqual(get_operator_description(node), "less than")

    def test_get_operator_description_and(self):
        node = ast.BoolOp(op=ast.And(), values=[ast.Constant(value=True), ast.Constant(value=False)])
        self.assertEqual(get_operator_description(node), "and")

    def test_get_operator_description_no_ops(self):
        node = ast.Name(id='a', ctx=ast.Load())
        self.assertIsNone(get_operator_description(node))

    def test_get_operator_description_unsupported_operator(self):
        # The function now handles unsupported operators gracefully by returning None.
        class CustomOp(ast.AST):
            _fields = ()
        node = ast.Compare(left=ast.Name(id='a', ctx=ast.Load()), ops=[CustomOp()], comparators=[ast.Constant(value=1)])
        self.assertIsNone(get_operator_description(node))

    def test_flatten_list_simple(self):
        self.assertEqual(flatten_list([1, 2, 3]), [1, 2, 3])

    def test_flatten_list_nested_single_element(self):
        self.assertEqual(flatten_list([1, [2], 3]), [1, 2, 3])

    def test_flatten_list_nested_multi_element(self):
        self.assertEqual(flatten_list([1, [2, 3], 4]), [1, 2, 3, 4])

    def test_flatten_list_mixed_nesting(self):
        self.assertEqual(flatten_list([1, [2, [3]], 4, [5, 6]]), [1, 2, 3, 4, 5, 6])

    def test_flatten_list_empty(self):
        self.assertEqual(flatten_list([]), [])

    def test_flatten_list_empty_nested(self):
        self.assertEqual(flatten_list([[], [[]]]), [])

    def test_flatten_list_non_list_elements(self):
        self.assertEqual(flatten_list([1, "a", True]), [1, "a", True])

if __name__ == '__main__':
    unittest.main()
