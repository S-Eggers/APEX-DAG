from __future__ import annotations

import numpy as np

class FeatureScaler:
    """z-score standardizer: (x - mean) / std per feature dimension."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, x: np.ndarray, eps: float = 1e-6) -> "FeatureScaler":
        x = np.asarray(x, dtype=np.float32)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        std = np.where(std < eps, 1.0, std)
        return cls(mean, std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / self.std

    def to_meta(self) -> dict:
        return {"feature_mean": self.mean.tolist(), "feature_std": self.std.tolist()}

    @classmethod
    def from_meta(cls, meta: dict) -> "FeatureScaler | None":
        if not meta or "feature_mean" not in meta or "feature_std" not in meta:
            return None
        return cls(np.asarray(meta["feature_mean"], dtype=np.float32),
                   np.asarray(meta["feature_std"], dtype=np.float32))
