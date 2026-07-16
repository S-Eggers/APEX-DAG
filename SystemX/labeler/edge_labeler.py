import abc

from SystemX.sca.cdf_ir import CDFIntermediateRepresentation


class EdgeLabeler(abc.ABC):
    @abc.abstractmethod
    def apply_labels(self, graph: CDFIntermediateRepresentation) -> None:
        pass
