from SystemX.pipeline.gat_labeling_pipeline_factory import GATLabelingPipelineFactory
from SystemX.pipeline.hgt_labeling_pipeline_factory import HGTLabelingPipelineFactory
from SystemX.pipeline.labeling_pipeline import LabelingPipeline
from SystemX.pipeline.llm_labeling_pipeline_factory import LLMLabelingPipelineFactory
from SystemX.pipeline.mlp_labeling_pipeline_factory import MLPLabelingPipelineFactory
from SystemX.pipeline.vamsa_labeling_pipeline_factory import VamsaLabelingPipelineFactory
from SystemX.pipeline.xgboost_labeling_pipeline_factory import XGBoostLabelingPipelineFactory


class LabelingPipelineFactory:
    @staticmethod
    def create(request_payload: dict | None = None, model: object | None = None, **kwargs: object) -> LabelingPipeline:
        if request_payload is None:
            request_payload = kwargs.get("request_payload", {})
        if model is None:
            model = kwargs.get("model")

        if request_payload.get("llmClassification"):
            config = kwargs.get("config", {})
            return LLMLabelingPipelineFactory.create(config)

        if request_payload.get("vamsaStaticClassification"):
            return VamsaLabelingPipelineFactory.create(request_payload)

        if request_payload.get("hgtClassification"):
            return HGTLabelingPipelineFactory.create(request_payload, model)

        if request_payload.get("mlpClassification"):
            return MLPLabelingPipelineFactory.create(request_payload, model)

        if request_payload.get("xgboostClassification"):
            return XGBoostLabelingPipelineFactory.create(request_payload, model)

        return GATLabelingPipelineFactory.create(request_payload, model or {})
