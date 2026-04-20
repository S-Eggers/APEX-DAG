import abc
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph

class EdgeLabeler(abc.ABC):
    @abc.abstractmethod
    def apply_labels(self, graph: PythonDataFlowGraph) -> None:
        pass