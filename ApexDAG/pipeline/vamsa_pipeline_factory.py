import os
from typing import Any

from ApexDAG.pipeline.vamsa_pipeline import VamsaPipeline
from ApexDAG.serializer.vamsa_serializer import VamsaMode, VamsaSerializer


class VamsaPipelineFactory:
    @staticmethod
    def create(request_payload: dict[str, Any]) -> VamsaPipeline:
        """
        Instantiates the VamsaPipeline with necessary serializers and file paths.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        kb_path = os.path.join(current_dir, "..", "vamsa", "data", "enhanced_annotation_kb.csv")
        mode = VamsaMode.WIR if request_payload.get("mode", 0) == 0 else VamsaMode.LINEAGE
        serializer = VamsaSerializer(mode)

        return VamsaPipeline(
            serializer=serializer,
            kb_csv_path=kb_path if os.path.exists(kb_path) else None,
        )
