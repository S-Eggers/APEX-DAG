from SystemX.pipeline.gat_lineage_pipeline_factory import GATLineagePipelineFactory
from SystemX.pipeline.hgt_lineage_pipeline_factory import HGTLineagePipelineFactory
from SystemX.pipeline.lineage_pipeline import LineagePipeline
from SystemX.pipeline.llm_lineage_pipeline_factory import LLMLineagePipelineFactory
from SystemX.pipeline.mlp_lineage_pipeline_factory import MLPLineagePipelineFactory
from SystemX.pipeline.vamsa_lineage_pipeline_factory import VamsaLineagePipelineFactory
from SystemX.pipeline.xgboost_lineage_pipeline_factory import XGBoostLineagePipelineFactory


class LineagePipelineFactory:
    @staticmethod
    def create(request_payload: dict | None = None, model: object | None = None, **kwargs: object) -> LineagePipeline:
        if request_payload is None:
            request_payload = kwargs.get("request_payload", {})
        if model is None:
            model = kwargs.get("model")

        if request_payload.get("llmClassification"):
            config = kwargs.get("config", {})
            return LLMLineagePipelineFactory.create(request_payload, config)

        if request_payload.get("vamsaStaticClassification"):
            return VamsaLineagePipelineFactory.create(request_payload)

        if request_payload.get("hgtClassification"):
            return HGTLineagePipelineFactory.create(request_payload, model)

        if request_payload.get("mlpClassification"):
            return MLPLineagePipelineFactory.create(request_payload, model)

        if request_payload.get("xgboostClassification"):
            return XGBoostLineagePipelineFactory.create(request_payload, model)

        return GATLineagePipelineFactory.create(request_payload, model or {})
