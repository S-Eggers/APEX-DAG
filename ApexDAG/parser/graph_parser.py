from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph


class GraphParser:
    def __init__(self, replace_dataflow: bool) -> None:
        self.replace_dataflow = replace_dataflow

    def parse(self, code: list) -> PythonDataFlowGraph:
        graph = PythonDataFlowGraph(replace_dataflow=self.replace_dataflow)
        graph.parse_cells(code)
        return graph
