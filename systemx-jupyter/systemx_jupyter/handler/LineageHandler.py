import logging

from SystemX.pipeline.hgt_lineage_pipeline_factory import HGTLineagePipelineFactory
from SystemX.pipeline.lineage_pipeline import LineagePipeline
from SystemX.pipeline.mlp_lineage_pipeline_factory import MLPLineagePipelineFactory
from SystemX.pipeline.vamsa_lineage_pipeline_factory import VamsaLineagePipelineFactory
from SystemX.pipeline.xgboost_lineage_pipeline_factory import XGBoostLineagePipelineFactory

from .SystemXBaseHandler import SystemXBaseHandler

logger = logging.getLogger(__name__)

_LINEAGE_FACTORIES = {
    "hgt": HGTLineagePipelineFactory,
    "mlp": MLPLineagePipelineFactory,
    "xgboost": XGBoostLineagePipelineFactory,
}

class LineageHandler(SystemXBaseHandler):
    @property
    def response_key(self) -> str:
        return "lineage_predictions"

    def create_pipeline(self, input_data: dict) -> LineagePipeline:
        backend: str = input_data.get("nnBackend", "hgt")
        preset: str = input_data.get("featurePreset", "standard")
        variant: str = (input_data.get("modelVariant") or "").strip()
        self.log.info("LineageHandler: nnBackend=%r featurePreset=%r modelVariant=%r", backend, preset, variant)

        if backend == "vamsa_static":
            return VamsaLineagePipelineFactory.create(input_data)

        if variant:
            family = variant.split("_", 1)[0]
            factory_family = family if family in _LINEAGE_FACTORIES else backend
            labeler = self.resolve_learned_labeler(factory_family, preset, variant_key=variant)
            return _LINEAGE_FACTORIES[factory_family].create(input_data, labeler)

        labeler = self.resolve_learned_labeler(backend, preset)
        return _LINEAGE_FACTORIES[backend].create(input_data, labeler)
