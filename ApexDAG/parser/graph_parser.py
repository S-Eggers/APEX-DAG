import abc

from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph


class GraphParser:
    def __init__(self, replace_dataflow: bool):
        self.replace_dataflow = replace_dataflow

    def parse(self, code: str) -> PythonDataFlowGraph:
        graph = PythonDataFlowGraph(replace_dataflow=self.replace_dataflow)
        graph.parse_code(code)
        return graph