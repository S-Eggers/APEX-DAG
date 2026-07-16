from SystemX.parser.sanitizer_mixin import IPythonSanitizerMixin
from SystemX.sca.py_ast_graph import PythonASTGraph

class ASTParser(IPythonSanitizerMixin):
    def parse(self, code: list) -> PythonASTGraph:
        graph = PythonASTGraph()

        sanitized_code = self.sanitize_ipython_cells(code)

        graph.parse_cells(sanitized_code)
        return graph
