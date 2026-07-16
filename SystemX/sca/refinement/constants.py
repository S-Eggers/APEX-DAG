import re
from typing import Final

ATTR_NODE_TYPE: Final[str] = "node_type"
ATTR_PREDICTED_LABEL: Final[str] = "predicted_label"
ATTR_RAW_CODE: Final[str] = "raw_code"
ATTR_EDGE_LABEL: Final[str] = "label"
ATTR_HAS_LEAKAGE: Final[str] = "has_leakage"
ATTR_LEAKAGE_CLASS: Final[str] = "leakage_class"
ATTR_LEAKAGE_FINDINGS: Final[str] = "leakage_findings"
ATTR_IS_DEAD_CODE: Final[str] = "is_dead_code"
ATTR_DOMAIN_NODE: Final[str] = "domain_node"

TAINT_FLAG: Final[str] = "tainted"

ATTR_PREDICTED_CONFIDENCE: Final[str] = "predicted_confidence"
ATTR_PREDICTED_MARGIN: Final[str] = "predicted_margin"

CONFIDENCE_OVERRIDE_THRESHOLD: Final[float] = 0.6

MARGIN_OVERRIDE_THRESHOLD: Final[float] = 0.12

def callee_name(code: str) -> str:
    """The method/attribute/function actually invoked - the identifier just before the first ( (or the last dotted token for a bare attribute access)."""
    head = str(code or "").strip().split("(", 1)[0]
    tokens = [t for t in re.split(r"[.\[]", head) if t.strip()]
    return tokens[-1].strip() if tokens else ""

def is_confident(node_data: dict, threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD) -> bool:
    """True if the labeler was confident about this node (don't overwrite)."""
    return float(node_data.get(ATTR_PREDICTED_CONFIDENCE, 1.0)) >= threshold

def is_overridable(
    node_data: dict,
    confidence_threshold: float = CONFIDENCE_OVERRIDE_THRESHOLD,
    margin_threshold: float = MARGIN_OVERRIDE_THRESHOLD,
) -> bool:
    """True if a runner-up reassignment rule may overwrite this node's label."""
    conf = float(node_data.get(ATTR_PREDICTED_CONFIDENCE, 1.0))
    margin = float(node_data.get(ATTR_PREDICTED_MARGIN, 1.0))
    return conf < confidence_threshold and margin < margin_threshold
