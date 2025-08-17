import unittest
import ast
from collections import defaultdict
from ApexDAG.sca.import_visitor import ImportVisitor

class TestImportVisitor(unittest.TestCase):

    def setUp(self):
        self.visitor = ImportVisitor()

    def test_initialization(self):
        self.assertEqual(self.visitor.imports, [])
        self.assertEqual(self.visitor.classes, [])
        self.assertEqual(self.visitor.functions, [])
        self.assertIsInstance(self.visitor.import_usage, defaultdict)
        self.assertIsInstance(self.visitor.import_counts, defaultdict)
        self.assertIsNone(self.visitor.current_class)

    def test_visit_import_simple(self):
        code = "import os\nimport sys"
        tree = ast.parse(code)
        self.visitor.visit(tree)
        self.assertEqual(self.visitor.imports, [('os', None), ('sys', None)])
        self.assertEqual(self.visitor.import_usage['os'], set())
        self.assertEqual(self.visitor.import_usage['sys'], set())

    def test_visit_import_as_alias(self):
        code = "import pandas as pd"
        tree = ast.parse(code)
        self.visitor.visit(tree)
        self.assertEqual(self.visitor.imports, [('pandas', 'pd')])
        self.assertEqual(self.visitor.import_usage['pd'], set())

    def test_visit_import_from_simple(self):
        code = "from collections import defaultdict"
        tree = ast.parse(code)
        self.visitor.visit(tree)
        self.assertEqual(self.visitor.imports, [('collections.defaultdict', None)])
        self.assertEqual(self.visitor.import_usage['defaultdict'], set())

    def test_visit_import_from_as_alias(self):
        code = "from numpy import array as arr"
        tree = ast.parse(code)
        self.visitor.visit(tree)
        self.assertEqual(self.visitor.imports, [('numpy.array', 'arr')])
        self.assertEqual(self.visitor.import_usage['arr'], set())

    def test_visit_class_def(self):
        code = "class MyClass:\n    pass"
        tree = ast.parse(code)
        self.visitor.visit(tree)
        self.assertEqual(self.visitor.classes, ['MyClass'])

    def test_visit_function_def_top_level(self):
        code = "def my_func():\n    pass"
        tree = ast.parse(code)
        self.visitor.visit(tree)
        self.assertEqual(self.visitor.functions, ['my_func'])

    def test_visit_function_def_in_class(self):
        code = "class MyClass:\n    def my_method(self):\n        pass"
        tree = ast.parse(code)
        self.visitor.visit(tree)
        self.assertEqual(self.visitor.functions, ['MyClass.my_method'])

    def test_visit_attribute(self):
        code = "import os\nprint(os.path)"
        tree = ast.parse(code)
        self.visitor.visit(tree)
        self.assertEqual(self.visitor.import_usage['os'], {'path', None})
        self.assertEqual(self.visitor.import_counts['os'], 2)

    def test_visit_name(self):
        code = "import sys\nprint(sys)"
        tree = ast.parse(code)
        self.visitor.visit(tree)
        self.assertEqual(self.visitor.import_usage['sys'], {None})
        self.assertEqual(self.visitor.import_counts['sys'], 1)

    def test_full_scenario(self):
        code = """
import os
from collections import Counter as Cnt

class MyProcessor:
    def __init__(self):
        self.data = []

    def process(self, item):
        self.data.append(item)
        if os.path.exists('file.txt'):
            print(Cnt([1,2,3]))

def main():
    processor = MyProcessor()
    processor.process('test')
    print(os.getcwd())
"""
        tree = ast.parse(code)
        self.visitor.visit(tree)

        self.assertEqual(self.visitor.imports, [('os', None), ('collections.Counter', 'Cnt')])
        self.assertEqual(self.visitor.classes, ['MyProcessor'])
        self.assertEqual(self.visitor.functions, ['MyProcessor.__init__', 'MyProcessor.process', 'main'])
        self.assertEqual(self.visitor.import_usage['os'], {'path', 'getcwd', None})
        self.assertEqual(self.visitor.import_counts['os'], 4)
        self.assertEqual(self.visitor.import_usage['Cnt'], {None})
        self.assertEqual(self.visitor.import_counts['Cnt'], 1)

if __name__ == '__main__':
    unittest.main()
