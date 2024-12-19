import ast
from ApexDAG.scripts import Notebook
from ApexDAG.scripts.ast.import_visitor import ImportVisitor

class NotebookStatMiner:
    def __init__(self, notebook: Notebook):
        self.notebook = notebook
        self.imports = []
        self.classes = []
        self.functions = []
        self.import_usage = {}
        self.import_counts = {}
        self.cells_with_imports = 0
    
    @staticmethod
    def convert_to_array(val):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return []
        
    def _process_results(self, import_visitor: ImportVisitor):
        if len(import_visitor.imports) > 0:
            self.cells_with_imports += 1
        self.imports.extend(import_visitor.imports)
        self.classes.extend(import_visitor.classes)
        self.functions.extend(import_visitor.functions)
        
    def _mine_import_usages(self):
        code = self.notebook.code()
        import_visitor = ImportVisitor()
        tree = ast.parse(code)
        import_visitor.visit(tree)
        self.import_usage = dict(import_visitor.import_usage)
        self.import_counts = dict(import_visitor.import_counts)
        
    def _mine_cell_data(self):
        for cell in self.notebook:
            code = cell[0]["code"]
            import_visitor = ImportVisitor()
            tree = ast.parse(code)
            import_visitor.visit(tree)
            self._process_results(import_visitor)

    def mine(self, greedy: bool = True) -> tuple:
        origin_window_size = self.notebook._cell_window_size
        self.notebook._cell_window_size = 1
        self.notebook.create_execution_graph(greedy=greedy)
        self._mine_cell_data()        
        self._mine_import_usages()
        
        self.notebook._cell_window_size = origin_window_size
        if origin_window_size > 1:
            self.notebook.create_execution_graph(greedy=greedy)
        
        return self.imports, self.import_usage, self.import_counts, self.classes, self.functions, self.cells_with_imports