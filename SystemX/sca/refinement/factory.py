from __future__ import annotations

from .constants import CONFIDENCE_OVERRIDE_THRESHOLD, MARGIN_OVERRIDE_THRESHOLD
from .engine import GraphRefiner
from .rules.analysis import DeadCodeEliminationRule, LeakageDetectionRule
from .rules.extraction import BaseTruthExtractionRule, StaticSeedingRule
from .rules.leakage import LeakageAnalysisRule
from .rules.propagation import (
    BackwardFeatureRule,
    BackwardSourcePropagationRule,
    DynamicTaintPropagationRule,
    ForwardDatasetPropagationRule,
    ForwardModelPropagationRule,
    NoneRule,
    RelevancePropagationRule,
)
from .rules.resolution import (
    ArtifactSerializationRule,
    EvaluationSinkRule,
    LiteralResolutionRule,
    MutualExclusionRule,
)

_MARGIN_GATED_RULES = (RelevancePropagationRule, BackwardSourcePropagationRule)
_CONFIDENCE_GATED_RULES = (StaticSeedingRule, EvaluationSinkRule, ArtifactSerializationRule)

def _instantiate(cls: type, confidence_threshold: float, margin_threshold: float):
    """Build a rule instance, injecting confidence/margin thresholds where supported."""
    if cls in _MARGIN_GATED_RULES:
        return cls(confidence_threshold=confidence_threshold, margin_threshold=margin_threshold)
    if cls in _CONFIDENCE_GATED_RULES:
        return cls(confidence_threshold=confidence_threshold)
    return cls()

_EXTRACTION_RULES = [BaseTruthExtractionRule, StaticSeedingRule]
_PROPAGATION_RULES = [RelevancePropagationRule, BackwardSourcePropagationRule, BackwardFeatureRule, ForwardDatasetPropagationRule, ForwardModelPropagationRule, DynamicTaintPropagationRule]
_RESOLUTION_RULES = [EvaluationSinkRule, ArtifactSerializationRule, LiteralResolutionRule, MutualExclusionRule]
_ANALYSIS_RULES = [LeakageDetectionRule, DeadCodeEliminationRule]

_ALL_RULE_CLASSES = _EXTRACTION_RULES + _PROPAGATION_RULES + _RESOLUTION_RULES + _ANALYSIS_RULES

def _refiner_without(
    *excluded_classes: type,
    confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD,
    margin_threshold: float = MARGIN_OVERRIDE_THRESHOLD,
) -> GraphRefiner:
    """Builds a GraphRefiner from all default rules except those listed."""
    excluded = set(excluded_classes)
    rules = [_instantiate(cls, confidence_threshold, margin_threshold)
             for cls in _ALL_RULE_CLASSES if cls not in excluded]
    return GraphRefiner(rules=rules)

def _refiner_from(
    *included_classes: type,
    confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD,
    margin_threshold: float = MARGIN_OVERRIDE_THRESHOLD,
) -> GraphRefiner:
    """Builds a GraphRefiner containing only the listed rule classes."""
    return GraphRefiner(rules=[_instantiate(cls, confidence_threshold, margin_threshold)
                               for cls in included_classes])

def create_default_refiner(
    confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD,
    margin_threshold: float = MARGIN_OVERRIDE_THRESHOLD,
) -> GraphRefiner:
    """Full 10-rule pipeline (standard system configuration)."""
    return _refiner_from(*_ALL_RULE_CLASSES,
                         confidence_threshold=confidence_threshold,
                         margin_threshold=margin_threshold)

def create_seeding_refiner(
    confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD,
    margin_threshold: float = MARGIN_OVERRIDE_THRESHOLD,
) -> GraphRefiner:
    """Extraction only - BaseTruth + StaticSeeding, no propagation or resolution."""
    return _refiner_from(*_EXTRACTION_RULES,
                         confidence_threshold=confidence_threshold,
                         margin_threshold=margin_threshold)

def create_empty_refiner() -> GraphRefiner:
    """Null-object refiner - no rules applied."""
    return GraphRefiner(rules=[NoneRule()])

def create_no_propagation_refiner(
    confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD,
    margin_threshold: float = MARGIN_OVERRIDE_THRESHOLD,
) -> GraphRefiner:
    """All rules EXCEPT the propagation group (BackwardFeature + DynamicTaintPropagation)."""
    return _refiner_without(*_PROPAGATION_RULES,
                            confidence_threshold=confidence_threshold,
                            margin_threshold=margin_threshold)

def create_no_resolution_refiner(
    confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD,
    margin_threshold: float = MARGIN_OVERRIDE_THRESHOLD,
) -> GraphRefiner:
    """All rules EXCEPT the resolution group (EvaluationSink, ArtifactSerialization, LiteralResolution, MutualExclusion)."""
    return _refiner_without(*_RESOLUTION_RULES,
                            confidence_threshold=confidence_threshold,
                            margin_threshold=margin_threshold)

def create_no_analysis_refiner(
    confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD,
    margin_threshold: float = MARGIN_OVERRIDE_THRESHOLD,
) -> GraphRefiner:
    """All rules EXCEPT the analysis group (LeakageDetection + DeadCodeElimination)."""
    return _refiner_without(*_ANALYSIS_RULES,
                            confidence_threshold=confidence_threshold,
                            margin_threshold=margin_threshold)

def create_extraction_and_propagation_refiner(
    confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD,
    margin_threshold: float = MARGIN_OVERRIDE_THRESHOLD,
) -> GraphRefiner:
    """Extraction + Propagation only - no resolution or analysis."""
    return _refiner_from(*_EXTRACTION_RULES, *_PROPAGATION_RULES,
                         confidence_threshold=confidence_threshold,
                         margin_threshold=margin_threshold)

def create_leakage_refiner(
    confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD,
    margin_threshold: float = MARGIN_OVERRIDE_THRESHOLD,
) -> GraphRefiner:
    """Full pipeline with the static leakage/anti-pattern analysis (D1-D5)."""
    rule_classes = [cls for cls in _ALL_RULE_CLASSES if cls is not LeakageDetectionRule]
    rule_classes.append(LeakageAnalysisRule)
    return _refiner_from(*rule_classes,
                         confidence_threshold=confidence_threshold,
                         margin_threshold=margin_threshold)
