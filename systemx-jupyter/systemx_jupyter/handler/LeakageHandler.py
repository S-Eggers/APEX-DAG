import json
import logging
from pathlib import Path

import tornado
from SystemX.pipeline.labeling_pipeline import LabelingPipeline
from SystemX.sca.refinement.factory import create_leakage_refiner

from ..policy.WorkspacePolicy import WorkspacePolicy
from .SystemXBaseHandler import SystemXBaseHandler
from .LabelingPredictHandler import LabelingPredictHandler
from .LeakageSaveHandler import leakage_sidecar_name

logger = logging.getLogger(__name__)

class LeakageHandler(LabelingPredictHandler):
    """Serves the leakage-annotated graph for the Leakage view."""

    @property
    def response_key(self) -> str:
        return "predictions"

    def create_pipeline(self, input_data: dict) -> LabelingPipeline:
        pipeline = super().create_pipeline(input_data)
        pipeline.refiner = create_leakage_refiner()
        return pipeline

    @tornado.web.authenticated
    def post(self) -> None:
        SystemXBaseHandler.post(self)

    def postprocess_results(self, analysis_results: object, input_data: dict) -> object:
        """Overlay detector pre-labels and any saved human gold leakage labels onto the graph."""
        if not isinstance(analysis_results, dict):
            return analysis_results

        elements = analysis_results.get("elements", [])

        if input_data.get("useGraphRefiner") and input_data.get("usePredictionForAnnotation"):
            for element in elements:
                data = element.get("data") if isinstance(element, dict) else None
                if isinstance(data, dict) and data.get("has_leakage") and data.get("leakage_class"):
                    data["leakage_gold"] = data["leakage_class"]

        filename = input_data.get("filename", "")
        base_path = input_data.get("base_path", "")
        if not (filename and base_path):
            return analysis_results

        try:
            workspace = WorkspacePolicy(Path.cwd(), base_path)
            sidecar = workspace.get_secure_annotation_path(leakage_sidecar_name(filename))
            if not sidecar.exists():
                return analysis_results
            with open(sidecar, encoding="utf-8") as f:
                gold = (json.load(f) or {}).get("gold", {})
        except (ValueError, PermissionError, OSError, json.JSONDecodeError) as e:
            self.log.warning("Could not load leakage gold sidecar: %s", e)
            return analysis_results

        if not isinstance(gold, dict) or not gold:
            return analysis_results

        for element in elements:
            data = element.get("data") if isinstance(element, dict) else None
            if isinstance(data, dict) and data.get("id") in gold:
                data["leakage_gold"] = gold[data["id"]]
        return analysis_results
