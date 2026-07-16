from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfusionMatrix:
    """Encapsulates TP, FP, FN and standard metric calculations for shared use."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0 else 0.0

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0 else 0.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        denominator = p + r
        return (2 * p * r) / denominator if denominator > 0 else 0.0
