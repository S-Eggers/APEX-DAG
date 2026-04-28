from ApexDAG.parser.sanitizer_mixin import IPythonSanitizerMixin
from ApexDAG.sca.py_ast_graph import PythonASTGraph


class ASTParser(IPythonSanitizerMixin):
    def parse(self, code: list) -> PythonASTGraph:
        graph = PythonASTGraph()

        # Inject the sanitization phase
        sanitized_code = self.sanitize_ipython_cells(code)

        graph.parse_cells(sanitized_code)
        return graph
