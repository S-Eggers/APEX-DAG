from ApexDAG.sca.py_ast_graph import PythonASTGraph


class ASTParser:
    def parse(self, code: list) -> PythonASTGraph:
        graph = PythonASTGraph()
        graph.parse_cells(code)
        return graph
