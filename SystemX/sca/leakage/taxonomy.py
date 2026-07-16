from __future__ import annotations

from typing import Final, TypedDict

from .analyzer import LeakageClass

class LeakageClassMeta(TypedDict):
    """Presentation + identity metadata for one leakage gold class."""

    key: str
    label: str
    category: str
    color: str
    description: str
    detector: str | None

LEAKAGE_GOLD_TAXONOMY: Final[list[LeakageClassMeta]] = [
    {
        "key": "clean",
        "label": "Clean (no leakage)",
        "category": "Leakage-free",
        "color": "#009E73",
        "description": "The operation is sound - no train/test contamination, target leakage, or evaluation flaw.",
        "detector": None,
    },
    {
        "key": LeakageClass.PREPROCESSING_BEFORE_SPLIT.value,
        "label": "Preprocess before split",
        "category": "Train/test contamination",
        "color": "#AA4E86",
        "description": "A transformer is fit on data later split into train/test - statistics leak from held-out rows.",
        "detector": "D1",
    },
    {
        "key": LeakageClass.TEST_IN_TRAIN.value,
        "label": "Test data in training",
        "category": "Train/test contamination",
        "color": "#D55E00",
        "description": "Held-out data is fed to a training op (fit / partial_fit).",
        "detector": "D3",
    },
    {
        "key": LeakageClass.TARGET_LEAKAGE.value,
        "label": "Target leakage",
        "category": "Target leakage",
        "color": "#C77E00",
        "description": "The feature matrix is derived from the target label, so the model sees the answer at training time.",
        "detector": "D2",
    },
    {
        "key": LeakageClass.METRIC_ON_TRAIN.value,
        "label": "Metric on training data",
        "category": "Evaluation flaw",
        "color": "#56B4E9",
        "description": "A performance metric is computed on training data, overstating generalization.",
        "detector": "D4",
    },
    {
        "key": LeakageClass.NO_HOLDOUT_EVALUATION.value,
        "label": "No held-out evaluation",
        "category": "Evaluation flaw",
        "color": "#6B7280",
        "description": "A model is fit but the notebook never carves off a test set.",
        "detector": "D5",
    },
    {
        "key": "uncertain",
        "label": "Uncertain",
        "category": "Unresolved",
        "color": "#9CA3AF",
        "description": "Not enough context to decide - revisit or escalate.",
        "detector": None,
    },
]

LEAKAGE_GOLD_BY_KEY: Final[dict[str, LeakageClassMeta]] = {
    entry["key"]: entry for entry in LEAKAGE_GOLD_TAXONOMY
}
