from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from SystemX.nn.data.v2.feature_extractor import FeatureGroup
from SystemX.sca.refinement.engine import GraphRefiner
from SystemX.sca.refinement.factory import (
    create_default_refiner,
    create_empty_refiner,
    create_extraction_and_propagation_refiner,
    create_leakage_refiner,
    create_no_analysis_refiner,
    create_no_propagation_refiner,
    create_no_resolution_refiner,
    create_seeding_refiner,
)

VAMSA_CONFIDENCE_THRESHOLD = 1.0
VAMSA_MARGIN_THRESHOLD = 1.0

class LabelerType(StrEnum):
    """Selects which labeler is wired into the evaluation pipeline."""

    V2_HGT_FASTTEXT = "v2_hgt_fasttext"
    """HGT + FastText (new default - lightweight, fast)."""

    V2_HGT_CODEBERT = "v2_hgt_codebert"
    """HGT + CodeBERT (ablation - 768-d transformer)."""

    V2_HGT_GRAPHCODEBERT = "v2_hgt_graphcodebert"
    """HGT + GraphCodeBERT (ablation - code-aware 768-d transformer)."""

    V2_MLP = "v2_mlp"
    """MLP on FastText + structural features (no graph attention)."""

    V2_XGBOOST = "v2_xgboost"
    """XGBoost on FastText + structural features."""

    V1_GAT = "v1_gat"
    """Legacy V1 GATv2 edge classifier (kept for backward comparison)."""

    VAMSA_STATIC = "vamsa_static"
    """Vamsa KB static lookup - no neural model."""

    LLM = "llm"
    """LLM labeler. Currently backed by a random-classifier placeholder
    (``RandomLabeler``) so the variant runs end-to-end with no training; to be
    replaced with a real LLM labeler later."""

class AblationVariant(StrEnum):
    STANDARD = "standard"
    """Full 10-rule refinement pipeline with V2 HGT+FastText (baseline)."""

    SEEDING_REFINER = "seeding_refiner"
    """Extraction group only (BaseTruth + StaticSeeding)."""

    EMPTY_REFINER = "empty_refiner"
    """No refinement - labeler predictions used verbatim."""

    NO_PROPAGATION = "no_propagation"
    """All rules except the propagation group (BackwardFeature + DynamicTaint)."""

    NO_RESOLUTION = "no_resolution"
    """All rules except the resolution group (EvaluationSink, ArtifactSerialization,
    LiteralResolution, MutualExclusion)."""

    NO_ANALYSIS = "no_analysis"
    """All rules except the analysis group (LeakageDetection + DeadCodeElimination)."""

    EXTRACTION_AND_PROPAGATION = "extraction_and_propagation"
    """Extraction + Propagation only (no resolution, no analysis)."""

    LEAKAGE_ANALYSIS = "leakage_analysis"
    """Full pipeline with the static leakage/anti-pattern analysis (D1-D5) swapped
    in for the legacy name-heuristic leakage rule. Emits per-class findings on the
    ``leakage_findings`` channel - the follow-up defect-detection study."""

    WITH_SUBGRAPH_HIGHLIGHT = "with_subgraph_highlight"
    """Standard refiner with highlight_relevant=True."""

    VAMSA_STATIC_BASELINE = "vamsa_static_baseline"
    """Pure Vamsa KB lookup - no neural model."""

    LLM_BASELINE = "llm_baseline"
    """LLM labeler baseline (no training). Placeholder random classifier for now."""

    LLM_GEMINI_A = "llm_gemini_a"
    """Real zero-shot Gemini hub classifier - model A (config: gemini_hub_a.yaml)."""

    LLM_GEMINI_B = "llm_gemini_b"
    """Real zero-shot Gemini hub classifier - model B (config: gemini_hub_b.yaml)."""

    LLM_GEMMA_31B = "llm_gemma_31b"
    """Real zero-shot Google Gemma 4 31B hub classifier (config: gemma-4-31b.yaml)."""

    LLM_GEMMA_26B = "llm_gemma_26b"
    """Real zero-shot Google Gemma 4 26B hub classifier (config: gemma-4-26b.yaml)."""

    LLM_GEMINI_CODE = "llm_gemini_code"
    """Coding-specialized zero-shot Gemini Pro hub classifier (config: gemini-code.yaml)."""

    LLM_GEMINI_CODE_RICH = "llm_gemini_code_rich"
    """Context-rich Gemini Pro hub classifier - same model as llm_gemini_code but each
    prompt carries the node's topological context (inputs/outputs/downstream consumers).
    Config: gemini-code-rich.yaml."""

    LLM_GEMINI_STRONG = "llm_gemini_strong"
    """STRONG LLM baseline - whole-notebook, few-shot, chain-of-thought. One structured
    call labels every operation of a notebook at once (full pipeline context), primed
    with worked examples and CoT reasoning. Config: gemini-code-rich.yaml."""

    LLM_GEMMA2_2B_LOCAL = "llm_gemma2_2b_local"
    """LOCAL zero-shot Gemma 2 (2B) hub classifier, served on-device via an
    OpenAI-compatible server (mlx_lm.server). Config: gemma2-2b-local.yaml."""

    LLM_GEMMA2_2B_LOCAL_STRONG = "llm_gemma2_2b_local_strong"
    """LOCAL STRONG (competitive) Gemma 2 (2B) - whole-notebook, few-shot, CoT. Same
    on-device model as llm_gemma2_2b_local. Config: gemma2-2b-local-strong.yaml."""

    LLM_GEMMA2_9B_LOCAL = "llm_gemma2_9b_local"
    """LOCAL zero-shot Gemma 2 (9B) hub classifier, served on-device via an
    OpenAI-compatible server (mlx_lm.server). Config: gemma2-9b-local.yaml."""

    LLM_GEMMA2_9B_LOCAL_STRONG = "llm_gemma2_9b_local_strong"
    """LOCAL STRONG (competitive) Gemma 2 (9B) - whole-notebook, few-shot, CoT. Same
    on-device model as llm_gemma2_9b_local. Config: gemma2-9b-local-strong.yaml."""

    LLM_MISTRAL_7B_LOCAL = "llm_mistral_7b_local"
    """LOCAL zero-shot Mistral 7B hub classifier, served on-device via an
    OpenAI-compatible server (mlx_lm.server). Small dense model (~4GB@4bit, fits a
    24GB Mac). Config: mistral-7b-local.yaml."""

    LLM_MISTRAL_7B_LOCAL_STRONG = "llm_mistral_7b_local_strong"
    """LOCAL STRONG (competitive) Mistral 7B - whole-notebook, few-shot, CoT. Same
    on-device model as llm_mistral_7b_local. Config: mistral-7b-local-strong.yaml."""

    LLM_GEMMA4_4B_LOCAL = "llm_gemma4_4b_local"
    """LOCAL zero-shot Gemma 4 (4B) hub classifier, served on-device via an
    OpenAI-compatible server (mlx_lm.server). Config: gemma4-4b-local.yaml."""

    LLM_GEMMA4_4B_LOCAL_STRONG = "llm_gemma4_4b_local_strong"
    """LOCAL STRONG (competitive) Gemma 4 (4B) - whole-notebook, few-shot, CoT. Same
    on-device model as llm_gemma4_4b_local. Config: gemma4-4b-local-strong.yaml."""

    LLM_GEMMA4_12B_LOCAL = "llm_gemma4_12b_local"
    """LOCAL zero-shot Gemma 4 (12B) hub classifier, served on-device via an
    OpenAI-compatible server (mlx_lm.server). Config: gemma4-12b-local.yaml."""

    LLM_GEMMA4_12B_LOCAL_STRONG = "llm_gemma4_12b_local_strong"
    """LOCAL STRONG (competitive) Gemma 4 (12B) - whole-notebook, few-shot, CoT. Same
    on-device model as llm_gemma4_12b_local. Config: gemma4-12b-local-strong.yaml."""

    LLM_QWEN3_4B_LOCAL = "llm_qwen3_4b_local"
    """LOCAL zero-shot Qwen3 (4B) hub classifier, served on-device via an
    OpenAI-compatible server (mlx_lm.server). Config: qwen3-4b-local.yaml."""

    LLM_QWEN3_4B_LOCAL_STRONG = "llm_qwen3_4b_local_strong"
    """LOCAL STRONG (competitive) Qwen3 (4B) - whole-notebook, few-shot, CoT. Same
    on-device model as llm_qwen3_4b_local. Config: qwen3-4b-local-strong.yaml."""

    LLM_QWEN25_CODER_7B_LOCAL = "llm_qwen25_coder_7b_local"
    """LOCAL zero-shot Qwen2.5-Coder (7B) hub classifier, served on-device via an
    OpenAI-compatible server (mlx_lm.server). Code-specialized dense model.
    Config: qwen25-coder-7b-local.yaml."""

    LLM_QWEN25_CODER_7B_LOCAL_STRONG = "llm_qwen25_coder_7b_local_strong"
    """LOCAL STRONG (competitive) Qwen2.5-Coder (7B) - whole-notebook, few-shot, CoT.
    Same on-device model as llm_qwen25_coder_7b_local. Config: qwen25-coder-7b-local-strong.yaml."""

    V1_GAT_BASELINE = "v1_gat_baseline"
    """Legacy V1 GATv2 edge labeler (for comparison with V2)."""

    V2_CODEBERT = "v2_codebert"
    """HGT with CodeBERT embedding (ablation vs FastText)."""

    V2_GRAPHCODEBERT = "v2_graphcodebert"
    """HGT with GraphCodeBERT embedding (ablation vs FastText)."""

    V2_MLP = "v2_mlp"
    """MLP: code_emb + degree + cell_pos + neighbor (STANDARD features)."""

    V2_XGBOOST = "v2_xgboost"
    """XGBoost: code_emb + degree + cell_pos + neighbor (STANDARD features)."""

    V2_MLP_FEAT_EMB_ONLY = "v2_mlp_feat_emb_only"
    """MLP: code embedding only - no structural or cell features.
    Tests whether the text signal alone suffices."""

    V2_MLP_FEAT_ALL = "v2_mlp_feat_all"
    """MLP: code + api_name + base_inputs embeddings + all structural.
    Full 929-d feature vector - upper bound for MLP."""

    V2_MLP_FEAT_API_LIB = "v2_mlp_feat_api_lib"
    """MLP: api_name + base_inputs embeddings only - no full code or structural.
    Tests whether method name + library context alone classify well."""

    V2_MLP_FEAT_STRUCT_ONLY = "v2_mlp_feat_struct_only"
    """MLP: degree + cell_pos + neighbor + lineage - zero text, 15-d structural only.
    Lower bound: can graph topology alone classify?"""

    V2_MLP_FEAT_API29 = "v2_mlp_feat_api29"
    """MLP: api_name embedding + all 29 non-embedding features (329-d) - ALL minus the
    code and library embeddings. Tests whether the cheap API-name vector plus the
    structural/meta features reaches the full 929-d accuracy."""

    V2_MLP_FEAT_API24 = "v2_mlp_feat_api24"
    """MLP: api29 minus centrality (324-d) - OOD win without the expensive centrality cost."""

    V2_XGBOOST_FEAT_ALL = "v2_xgboost_feat_all"
    """XGBoost with all feature groups (929-d) - best possible XGBoost."""

    V2_XGBOOST_FEAT_EMB_ONLY = "v2_xgboost_feat_emb_only"
    """XGBoost: code embedding only (300-d)."""

    V2_XGBOOST_FEAT_API_LIB = "v2_xgboost_feat_api_lib"
    """XGBoost: api_name + base_inputs embeddings only (600-d)."""

    V2_XGBOOST_FEAT_STRUCT_ONLY = "v2_xgboost_feat_struct_only"
    """XGBoost: pure graph topology only (no text, no API-name anchors)."""

    V2_XGBOOST_FEAT_EMB_RICH = "v2_xgboost_feat_emb_rich"
    """XGBoost: former embedding-heavy standard - code+api+lib embeddings, library
    category and API-name anchors (924-d). Ablation vs the new code+structure
    standard: quantifies the in-distribution gain that fails to transfer OOD."""

    V2_XGBOOST_FEAT_API29 = "v2_xgboost_feat_api29"
    """XGBoost: api_name embedding + all 29 non-embedding features (329-d) - ALL minus
    the code and library embeddings."""

    V2_XGBOOST_FEAT_API24 = "v2_xgboost_feat_api24"
    """XGBoost: api29 minus centrality (324-d)."""

    V2_HGT_FEAT_EMB_ONLY = "v2_hgt_feat_emb_only"
    """HGT: code embedding only (300-d) on operation nodes."""

    V2_HGT_FEAT_ALL = "v2_hgt_feat_all"
    """HGT: all feature groups (929-d) on operation nodes - upper bound."""

    V2_HGT_FEAT_API_LIB = "v2_hgt_feat_api_lib"
    """HGT: api_name + base_inputs embeddings only (600-d) on operation nodes."""

    V2_HGT_FEAT_STRUCT_ONLY = "v2_hgt_feat_struct_only"
    """HGT: degree + cell_pos + neighbor + lineage (15-d structural only) on operation nodes.
    The GNN still sees the full graph structure, so this isolates the
    contribution of node-level text features vs. message passing."""

    V2_HGT_FEAT_API29 = "v2_hgt_feat_api29"
    """HGT: api_name embedding + all 29 non-embedding features (329-d) on operation
    nodes - ALL minus the code and library embeddings."""

    V2_HGT_FEAT_API24 = "v2_hgt_feat_api24"
    """HGT: api29 minus centrality (324-d) on operation nodes."""

    V2_MLP_UNDERSAMPLE = "v2_mlp_undersample"
    """MLP, STANDARD features, random under-sampling to the minority class."""

    V2_MLP_CLASSWEIGHT = "v2_mlp_classweight"
    """MLP, STANDARD features, inverse-frequency class-weighted loss."""

    V2_XGBOOST_UNDERSAMPLE = "v2_xgboost_undersample"
    """XGBoost, STANDARD features, random under-sampling to the minority class."""

    V2_XGBOOST_CLASSWEIGHT = "v2_xgboost_classweight"
    """XGBoost, STANDARD features, inverse-frequency per-sample weights."""

    V2_HGT_CW_NONE = "v2_hgt_cw_none"
    """HGT, STANDARD features, NO class weighting (vs the sqrt_inverse default in `standard`)."""

    V2_HGT_CW_BALANCED = "v2_hgt_cw_balanced"
    """HGT, STANDARD features, full inverse-frequency (balanced) class weighting."""

@dataclass
class AblationConfig:
    """Complete configuration for one ablation experiment."""

    refiner: GraphRefiner
    highlight_relevant: bool = False
    labeler_type: LabelerType = LabelerType.V2_HGT_FASTTEXT
    feature_groups: FeatureGroup = FeatureGroup.STANDARD
    llm_config_path: str | None = None
    llm_rich_context: bool = False
    llm_strong: bool = False

    @property
    def use_vamsa_static_labeler(self) -> bool:
        return self.labeler_type == LabelerType.VAMSA_STATIC

CONFIG_REGISTRY: dict[AblationVariant, AblationConfig] = {
    AblationVariant.STANDARD: AblationConfig(refiner=create_default_refiner()),
    AblationVariant.SEEDING_REFINER: AblationConfig(refiner=create_seeding_refiner(), feature_groups=FeatureGroup.API24),
    AblationVariant.EMPTY_REFINER: AblationConfig(refiner=create_empty_refiner(), feature_groups=FeatureGroup.API24),
    AblationVariant.NO_PROPAGATION: AblationConfig(refiner=create_no_propagation_refiner(), feature_groups=FeatureGroup.API24),
    AblationVariant.NO_RESOLUTION: AblationConfig(refiner=create_no_resolution_refiner(), feature_groups=FeatureGroup.API24),
    AblationVariant.NO_ANALYSIS: AblationConfig(refiner=create_no_analysis_refiner(), feature_groups=FeatureGroup.API24),
    AblationVariant.EXTRACTION_AND_PROPAGATION: AblationConfig(refiner=create_extraction_and_propagation_refiner(), feature_groups=FeatureGroup.API24),
    AblationVariant.LEAKAGE_ANALYSIS: AblationConfig(refiner=create_leakage_refiner()),
    AblationVariant.WITH_SUBGRAPH_HIGHLIGHT: AblationConfig(
        refiner=create_default_refiner(),
        highlight_relevant=True,
    ),
    AblationVariant.VAMSA_STATIC_BASELINE: AblationConfig(
        refiner=create_default_refiner(
            confidence_threshold=VAMSA_CONFIDENCE_THRESHOLD,
            margin_threshold=VAMSA_MARGIN_THRESHOLD,
        ),
        labeler_type=LabelerType.VAMSA_STATIC,
    ),
    AblationVariant.LLM_BASELINE: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
    ),
    AblationVariant.LLM_GEMINI_A: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemini_hub_a.yaml",
    ),
    AblationVariant.LLM_GEMINI_B: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemini_hub_b.yaml",
    ),
    AblationVariant.LLM_GEMMA_31B: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemma-4-31b.yaml",
    ),
    AblationVariant.LLM_GEMMA_26B: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemma-4-26b.yaml",
    ),
    AblationVariant.LLM_GEMINI_CODE: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemini-code.yaml",
    ),
    AblationVariant.LLM_GEMINI_CODE_RICH: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemini-code-rich.yaml",
        llm_rich_context=True,
    ),
    AblationVariant.LLM_GEMINI_STRONG: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemini-strong.yaml",
        llm_strong=True,
    ),
    AblationVariant.LLM_GEMMA2_2B_LOCAL: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemma2-2b-local.yaml",
    ),
    AblationVariant.LLM_GEMMA2_2B_LOCAL_STRONG: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemma2-2b-local-strong.yaml",
        llm_strong=True,
    ),
    AblationVariant.LLM_GEMMA2_9B_LOCAL: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemma2-9b-local.yaml",
    ),
    AblationVariant.LLM_GEMMA2_9B_LOCAL_STRONG: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemma2-9b-local-strong.yaml",
        llm_strong=True,
    ),
    AblationVariant.LLM_MISTRAL_7B_LOCAL: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/mistral-7b-local.yaml",
    ),
    AblationVariant.LLM_MISTRAL_7B_LOCAL_STRONG: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/mistral-7b-local-strong.yaml",
        llm_strong=True,
    ),
    AblationVariant.LLM_GEMMA4_4B_LOCAL: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemma4-4b-local.yaml",
    ),
    AblationVariant.LLM_GEMMA4_4B_LOCAL_STRONG: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemma4-4b-local-strong.yaml",
        llm_strong=True,
    ),
    AblationVariant.LLM_GEMMA4_12B_LOCAL: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemma4-12b-local.yaml",
    ),
    AblationVariant.LLM_GEMMA4_12B_LOCAL_STRONG: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/gemma4-12b-local-strong.yaml",
        llm_strong=True,
    ),
    AblationVariant.LLM_QWEN3_4B_LOCAL: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/qwen3-4b-local.yaml",
    ),
    AblationVariant.LLM_QWEN3_4B_LOCAL_STRONG: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/qwen3-4b-local-strong.yaml",
        llm_strong=True,
    ),
    AblationVariant.LLM_QWEN25_CODER_7B_LOCAL: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/qwen25-coder-7b-local.yaml",
    ),
    AblationVariant.LLM_QWEN25_CODER_7B_LOCAL_STRONG: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.LLM,
        llm_config_path="SystemX/llm/config/qwen25-coder-7b-local-strong.yaml",
        llm_strong=True,
    ),
    AblationVariant.V1_GAT_BASELINE: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V1_GAT,
    ),
    AblationVariant.V2_CODEBERT: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_HGT_CODEBERT,
    ),
    AblationVariant.V2_GRAPHCODEBERT: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_HGT_GRAPHCODEBERT,
    ),
    AblationVariant.V2_MLP: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_MLP,
        feature_groups=FeatureGroup.STANDARD,
    ),
    AblationVariant.V2_XGBOOST: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_XGBOOST,
        feature_groups=FeatureGroup.STANDARD,
    ),
    AblationVariant.V2_MLP_FEAT_EMB_ONLY: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_MLP,
        feature_groups=FeatureGroup.CODE_EMB,
    ),
    AblationVariant.V2_MLP_FEAT_ALL: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_MLP,
        feature_groups=FeatureGroup.ALL,
    ),
    AblationVariant.V2_MLP_FEAT_API_LIB: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_MLP,
        feature_groups=FeatureGroup.API_EMB | FeatureGroup.LIB_EMB,
    ),
    AblationVariant.V2_MLP_FEAT_STRUCT_ONLY: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_MLP,
        feature_groups=FeatureGroup.STRUCTURAL_ONLY,
    ),
    AblationVariant.V2_MLP_FEAT_API29: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_MLP,
        feature_groups=FeatureGroup.API29,
    ),
    AblationVariant.V2_MLP_FEAT_API24: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_MLP,
        feature_groups=FeatureGroup.API24,
    ),
    AblationVariant.V2_XGBOOST_FEAT_ALL: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_XGBOOST,
        feature_groups=FeatureGroup.ALL,
    ),
    AblationVariant.V2_XGBOOST_FEAT_EMB_ONLY: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_XGBOOST,
        feature_groups=FeatureGroup.CODE_EMB,
    ),
    AblationVariant.V2_XGBOOST_FEAT_API_LIB: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_XGBOOST,
        feature_groups=FeatureGroup.API_EMB | FeatureGroup.LIB_EMB,
    ),
    AblationVariant.V2_XGBOOST_FEAT_STRUCT_ONLY: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_XGBOOST,
        feature_groups=FeatureGroup.STRUCTURAL_ONLY,
    ),
    AblationVariant.V2_XGBOOST_FEAT_EMB_RICH: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_XGBOOST,
        feature_groups=FeatureGroup.EMB_RICH,
    ),
    AblationVariant.V2_XGBOOST_FEAT_API29: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_XGBOOST,
        feature_groups=FeatureGroup.API29,
    ),
    AblationVariant.V2_XGBOOST_FEAT_API24: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_XGBOOST,
        feature_groups=FeatureGroup.API24,
    ),
    AblationVariant.V2_HGT_FEAT_EMB_ONLY: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_HGT_FASTTEXT,
        feature_groups=FeatureGroup.CODE_EMB,
    ),
    AblationVariant.V2_HGT_FEAT_ALL: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_HGT_FASTTEXT,
        feature_groups=FeatureGroup.ALL,
    ),
    AblationVariant.V2_HGT_FEAT_API_LIB: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_HGT_FASTTEXT,
        feature_groups=FeatureGroup.API_EMB | FeatureGroup.LIB_EMB,
    ),
    AblationVariant.V2_HGT_FEAT_STRUCT_ONLY: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_HGT_FASTTEXT,
        feature_groups=FeatureGroup.STRUCTURAL_ONLY,
    ),
    AblationVariant.V2_HGT_FEAT_API29: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_HGT_FASTTEXT,
        feature_groups=FeatureGroup.API29,
    ),
    AblationVariant.V2_HGT_FEAT_API24: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_HGT_FASTTEXT,
        feature_groups=FeatureGroup.API24,
    ),
    AblationVariant.V2_MLP_UNDERSAMPLE: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_MLP,
        feature_groups=FeatureGroup.STANDARD,
    ),
    AblationVariant.V2_MLP_CLASSWEIGHT: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_MLP,
        feature_groups=FeatureGroup.STANDARD,
    ),
    AblationVariant.V2_XGBOOST_UNDERSAMPLE: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_XGBOOST,
        feature_groups=FeatureGroup.STANDARD,
    ),
    AblationVariant.V2_XGBOOST_CLASSWEIGHT: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_XGBOOST,
        feature_groups=FeatureGroup.STANDARD,
    ),
    AblationVariant.V2_HGT_CW_NONE: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_HGT_FASTTEXT,
        feature_groups=FeatureGroup.STANDARD,
    ),
    AblationVariant.V2_HGT_CW_BALANCED: AblationConfig(
        refiner=create_default_refiner(),
        labeler_type=LabelerType.V2_HGT_FASTTEXT,
        feature_groups=FeatureGroup.STANDARD,
    ),
}

POLICY_REGISTRY: dict[AblationVariant, GraphRefiner] = {v: cfg.refiner for v, cfg in CONFIG_REGISTRY.items()}
