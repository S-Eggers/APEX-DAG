from typing import Dict, Any
from ApexDAG.sca.import_visitor import ImportVisitor
from ApexDAG.sca.complexity_visitor import ComplexityVisitor
from ApexDAG.pipeline.environment_pipeline import EnvironmentPipeline
from ApexDAG.serializer.environment_serializer import EnvironmentSerializer

class EnvironmentPipelineFactory:
    @staticmethod
    def create(request_payload: Dict[str, Any] = None) -> EnvironmentPipeline:
        return EnvironmentPipeline(
            serializer=EnvironmentSerializer(),
            import_visitor_cls=ImportVisitor,
            complexity_visitor_cls=ComplexityVisitor
        )