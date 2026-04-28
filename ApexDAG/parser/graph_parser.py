from ApexDAG.parser.sanitizer_mixin import IPythonSanitizerMixin
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph


class GraphParser(IPythonSanitizerMixin):
    def __init__(self, replace_dataflow: bool) -> None:
        self.replace_dataflow = replace_dataflow

    def parse(self, code: list) -> PythonDataFlowGraph:
        graph = PythonDataFlowGraph(replace_dataflow=self.replace_dataflow)

        sanitized_code = self.sanitize_ipython_cells(code)

        graph.parse_cells(sanitized_code)
        return graph
