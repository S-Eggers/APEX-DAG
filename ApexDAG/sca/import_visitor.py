import ast
from collections import defaultdict


class ImportVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports = []
        self.classes = []
        self.functions = []
        self.import_usage = defaultdict(set)
        self.import_counts = defaultdict(int)
        self.current_class = None

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports.append((alias.name, alias.asname))
            self.import_usage[name] = set()
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            import_name = f"{module}.{alias.name}"
            self.imports.append((import_name, alias.asname))
            self.import_usage[name] = set()
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        function_name = node.name
        if self.current_class:
            function_name = f"{self.current_class}.{function_name}"
        self.functions.append(function_name)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            if node.value.id in self.import_usage:
                self.import_usage[node.value.id].add(node.attr)
                self.import_counts[node.value.id] += 1
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in self.import_usage:
            self.import_usage[node.id].add(None)  # None means the whole module is used
            self.import_counts[node.id] += 1
        self.generic_visit(node)
