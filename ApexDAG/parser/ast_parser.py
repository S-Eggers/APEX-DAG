from ApexDAG.sca.py_ast_graph import PythonASTGraph

class ASTParser:
    def parse(self, code: str) -> PythonASTGraph:
        graph = PythonASTGraph()
        graph.parse_code(code)
        return graph