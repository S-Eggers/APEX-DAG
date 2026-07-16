from abc import ABC, abstractmethod
from typing import Any, Protocol


class Pipeline(ABC):
    @abstractmethod
    def execute(self, input_data: list) -> dict[str, Any]: ...


class PipelineFactory(Protocol):
    @staticmethod
    def create(**kwargs: object) -> "Pipeline": ...
