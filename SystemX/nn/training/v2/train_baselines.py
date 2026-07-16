import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from SystemX.nn.data.v2.fasttext_embedding import FastTextEmbeddingV2
from SystemX.nn.data.v2.feature_extractor import FEATURE_PRESETS, ComputeHubFeatureExtractor, FeatureGroup
from SystemX.nn.data.v2.feature_scaler import FeatureScaler
from SystemX.nn.models.v2.mlp import ComputeHubMLP
from SystemX.nn.training.v2.data_utils import annotation_to_networkx as _annotation_to_networkx
from SystemX.nn.training.v2.trainer import MLPTrainer
from SystemX.sca.constants import DOMAIN_EDGE_TYPES
from SystemX.util.logger import configure_systemx_logger

configure_systemx_logger()
logger = logging.getLogger("SystemX.nn.training.v2.train_baselines")

def _collect_dataset(
    ann_paths: list[Path],
    extractor: ComputeHubFeatureExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    all_X, all_y = [], []

    for ann_path in ann_paths:
        try:
            with open(ann_path, encoding="utf-8") as f:
                raw = json.load(f)

            elements = raw if isinstance(raw, list) else raw.get("elements", [])
            nx_G = _annotation_to_networkx(elements)

            x_node, _, labels = extractor.extract(nx_G)
            valid = np.array(labels) != -1
            if valid.sum() == 0:
                continue

            all_X.append(x_node[valid])
            all_y.append(np.array(labels)[valid])

        except Exception as e:
            logger.warning("Skipping %s: %s", ann_path.name, e)

    if not all_X:
        return np.empty((0, extractor.feature_dim)), np.empty((0,))

    return np.concatenate(all_X), np.concatenate(all_y)

def _undersample(x: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Random under-sampling: cut every class down to the minority-class count."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y.astype(int), return_counts=True)
    target = int(counts.min())
    keep = []
    for c in classes:
        idx = np.where(y.astype(int) == c)[0]
        if len(idx) > target:
            idx = rng.choice(idx, target, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    logger.info("Under-sampled %d -> %d nodes (%d classes × %d).", len(y), len(keep), len(classes), target)
    return x[keep], y[keep]

def _class_weight_vector(y: np.ndarray, num_classes: int, max_weight: float = 5.0) -> np.ndarray:
    """Per-class inverse-frequency weights, normalized to mean 1 and clipped."""
    counts = np.bincount(y.astype(int), minlength=num_classes).astype(float)
    total = counts.sum()
    raw = np.array([(total / (num_classes * c)) if c > 0 else 1.0 for c in counts])
    raw = raw / raw.mean()
    return np.minimum(raw, max_weight)

def train_mlp(x: np.ndarray, y: np.ndarray, epochs: int, output_dir: Path, groups: FeatureGroup = FeatureGroup.STANDARD,
              class_weights: np.ndarray | None = None) -> None:
    num_classes = len(DOMAIN_EDGE_TYPES)
    logger.info("Training MLP: %d samples, %d features, %d classes.", len(x), x.shape[1], num_classes)

    scaler = FeatureScaler.fit(x)
    x = scaler.transform(x)

    model = ComputeHubMLP(in_features=x.shape[1], num_classes=num_classes)
    weight_t = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
    if weight_t is not None:
        logger.info("MLP class weights (normalized): %s", [round(float(w), 3) for w in weight_t])
    criterion = nn.CrossEntropyLoss(weight=weight_t)
    trainer = MLPTrainer(
        model=model,
        criterion=criterion,
        learning_rate=1e-3,
        log_dir=str(output_dir / "runs" / "mlp"),
    )

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    trainer.log_model_graph(x_t)

    log_every = max(1, epochs // 5)
    for epoch in range(1, epochs + 1):
        loss, acc = trainer.train_epoch(x_t, y_t, epoch)
        if epoch % log_every == 0 or epoch == epochs:
            logger.info("MLP epoch %d/%d  loss=%.4f  acc=%.3f", epoch, epochs, loss, acc)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"mlp_baseline_{timestamp}.pt"
    trainer.save_cpu_checkpoint(
        str(path),
        meta={
            "in_features": x.shape[1],
            "hidden": 256,
            "num_classes": num_classes,
            "feature_groups": groups.value,
            "use_batchnorm": bool(model.use_batchnorm),
            **scaler.to_meta(),
        },
    )
    logger.info("MLP checkpoint saved to %s", path)

def train_xgboost(x: np.ndarray, y: np.ndarray, output_dir: Path, class_weights: np.ndarray | None = None) -> None:
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost is required: pip install xgboost") from e

    num_classes = len(DOMAIN_EDGE_TYPES)
    logger.info("Training XGBoost: %d samples.", len(x))
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    sample_weight = class_weights[y.astype(int)] if class_weights is not None else None
    if sample_weight is not None:
        logger.info("XGBoost class weights (normalized): %s", [round(float(w), 3) for w in class_weights])
    model.fit(x, y.astype(int), sample_weight=sample_weight)

    from collections import Counter

    preds = model.predict(x)
    acc = (preds == y.astype(int)).mean()
    dist = Counter(preds.tolist())
    logger.info("XGBoost training accuracy: %.3f  pred distribution: %s", acc, dict(dist))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"xgboost_baseline_{timestamp}.json"
    model.save_model(str(path))
    logger.info("XGBoost model saved to %s", path)

_FEATURE_GROUP_MAP = FEATURE_PRESETS

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP/XGBoost baselines (V2)")
    parser.add_argument("--annotations_dir", required=True)
    parser.add_argument("--output_dir", default="./checkpoints/v2")
    parser.add_argument("--model", default="both", choices=["mlp", "xgboost", "both"])
    parser.add_argument("--epochs", type=int, default=50, help="MLP epochs")
    parser.add_argument(
        "--features",
        default="standard",
        choices=list(_FEATURE_GROUP_MAP),
        help="Feature group preset (see FeatureGroup in feature_extractor.py)",
    )
    parser.add_argument("--train_list", default=None, help="Optional file listing the fold's training annotation filenames (one per line)")
    parser.add_argument(
        "--sampling",
        default="none",
        choices=["none", "undersample", "class_weight"],
        help="Class-imbalance handling: none | undersample (random under-sample to minority) | class_weight (inverse-freq weighting)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = _FEATURE_GROUP_MAP[args.features]
    extractor = ComputeHubFeatureExtractor(FastTextEmbeddingV2(), groups=groups)
    logger.info("Feature set: %s", extractor.description)
    annotations_dir = Path(args.annotations_dir)
    if args.train_list and Path(args.train_list).exists():
        names = [ln.strip() for ln in Path(args.train_list).read_text().splitlines() if ln.strip()]
        ann_paths = [annotations_dir / n for n in names]
    else:
        ann_paths = sorted(annotations_dir.glob("*.json"))
    logger.info("Found %d annotation files.", len(ann_paths))

    x_data, y_data = _collect_dataset(ann_paths, extractor)
    if len(x_data) == 0:
        logger.error("No labelled CALL nodes found. Check annotation format.")
        return

    from collections import Counter

    logger.info(
        "Dataset: %d labelled CALL nodes, %d features.  Label distribution: %s",
        len(x_data),
        x_data.shape[1],
        dict(Counter(y_data.astype(int).tolist())),
    )

    num_classes = len(DOMAIN_EDGE_TYPES)
    class_weights = None
    if args.sampling == "undersample":
        x_data, y_data = _undersample(x_data, y_data)
    elif args.sampling == "class_weight":
        class_weights = _class_weight_vector(y_data, num_classes)
    logger.info("Sampling strategy: %s", args.sampling)

    if args.model in ("mlp", "both"):
        train_mlp(x_data, y_data, args.epochs, output_dir, groups, class_weights=class_weights)
    if args.model in ("xgboost", "both"):
        train_xgboost(x_data, y_data, output_dir, class_weights=class_weights)

if __name__ == "__main__":
    main()
