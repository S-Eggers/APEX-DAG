from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from SystemX.nn.data.v2.feature_extractor import (
    ComputeHubFeatureExtractor,
    FeatureGroup,
    infer_feature_groups,
)
from SystemX.nn.data.v2.feature_scaler import FeatureScaler
from SystemX.nn.models.v2.mlp import ComputeHubMLP
from SystemX.nn.training.v2.data_utils import annotation_to_networkx as _annotation_to_networkx
from SystemX.nn.training.v2.train_baselines import _collect_dataset
from SystemX.nn.training.v2.trainer import SystemXOnlineTrainer, MLPTrainer
from SystemX.sca.constants import DOMAIN_EDGE_TYPES

logger = logging.getLogger("SystemX.nn.training.v2.finetune")

ProgressCB = Callable[[int, str], None]

_SUFFIX = {"hgt": "pt", "mlp": "pt", "xgboost": "json"}

def _load_annotation_paths(annotations_dir: Path, train_list: Path | None) -> list[Path]:
    if train_list is not None and train_list.exists():
        names = [ln.strip() for ln in train_list.read_text().splitlines() if ln.strip()]
        return [annotations_dir / n for n in names]
    return sorted(annotations_dir.glob("*.json"))

def _emit(cb: ProgressCB | None, pct: int, msg: str) -> None:
    logger.info("[finetune] %d%% %s", pct, msg)
    if cb is not None:
        try:
            cb(pct, msg)
        except Exception:
            logger.warning("[finetune] progress callback failed", exc_info=True)

def finetune_variant(
    base_ckpt: str | Path,
    family: str,
    annotations_dir: str | Path,
    output_dir: str | Path,
    *,
    out_stem: str,
    epochs: int = 5,
    lr: float = 5e-4,
    xgb_rounds: int = 60,
    progress_cb: ProgressCB | None = None,
    train_list: str | Path | None = None,
    embedding_model: object | None = None,
) -> tuple[Path, dict]:
    """Fine-tune *base_ckpt* on the annotations in *annotations_dir*."""
    base_ckpt = Path(base_ckpt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir = Path(annotations_dir)
    tl = Path(train_list) if train_list else None
    ann_paths = _load_annotation_paths(annotations_dir, tl)
    if not ann_paths:
        raise ValueError(f"No annotation files found in {annotations_dir}")

    new_path = output_dir / f"{out_stem}.{_SUFFIX[family]}"

    _emit(progress_cb, 2, f"Loading {len(ann_paths)} annotation files")

    if family == "xgboost":
        return _finetune_xgboost(
            base_ckpt, ann_paths, new_path, xgb_rounds=xgb_rounds,
            embedding_model=embedding_model, progress_cb=progress_cb,
        )
    if family == "mlp":
        return _finetune_mlp(
            base_ckpt, ann_paths, new_path, epochs=epochs, lr=lr,
            embedding_model=embedding_model, progress_cb=progress_cb,
        )
    if family == "hgt":
        return _finetune_hgt(
            base_ckpt, ann_paths, new_path, epochs=epochs, lr=lr,
            embedding_model=embedding_model, progress_cb=progress_cb,
        )
    raise ValueError(f"Unknown model family {family!r}; expected hgt | mlp | xgboost.")

def _finetune_xgboost(
    base_ckpt: Path,
    ann_paths: list[Path],
    new_path: Path,
    *,
    xgb_rounds: int,
    embedding_model: object | None,
    progress_cb: ProgressCB | None,
) -> tuple[Path, dict]:
    try:
        import xgboost as xgb
    except ImportError as e:
        raise ImportError("xgboost is required: pip install xgboost") from e

    num_classes = len(DOMAIN_EDGE_TYPES)

    base = xgb.XGBClassifier()
    base.load_model(str(base_ckpt))
    groups = infer_feature_groups(int(base.n_features_in_))
    logger.info("[finetune] xgboost base preset inferred: %s", groups)

    extractor = ComputeHubFeatureExtractor(embedding=embedding_model, groups=groups)
    _emit(progress_cb, 20, "Extracting features")
    x, y = _collect_dataset(ann_paths, extractor)
    if len(x) == 0:
        raise ValueError("No labelled CALL nodes found in the annotation corpus.")

    _emit(progress_cb, 50, f"Boosting +{xgb_rounds} rounds on {len(x)} nodes")
    model = xgb.XGBClassifier(
        n_estimators=xgb_rounds,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    model.fit(x, y.astype(int), xgb_model=str(base_ckpt))
    model.save_model(str(new_path))

    preds = model.predict(x)
    acc = float((preds == y.astype(int)).mean())
    metrics = _metrics(len(ann_paths), x, y, acc, final_loss=None)
    _emit(progress_cb, 100, f"Done (train acc {acc:.3f})")
    logger.info("[finetune] xgboost checkpoint saved to %s", new_path)
    return new_path, metrics

def _finetune_mlp(
    base_ckpt: Path,
    ann_paths: list[Path],
    new_path: Path,
    *,
    epochs: int,
    lr: float,
    embedding_model: object | None,
    progress_cb: ProgressCB | None,
) -> tuple[Path, dict]:
    num_classes = len(DOMAIN_EDGE_TYPES)
    state = torch.load(str(base_ckpt), map_location="cpu", weights_only=True)
    sd = state["model_state_dict"]
    in_features = state.get("in_features", 302)

    use_bn = state.get("use_batchnorm")
    if use_bn is None:
        use_bn = any(k.endswith("running_mean") for k in sd)
    model = ComputeHubMLP(
        in_features=in_features,
        hidden=state.get("hidden", 256),
        num_classes=state.get("num_classes", num_classes),
        use_batchnorm=bool(use_bn),
    )
    model.load_state_dict(sd)

    scaler = FeatureScaler.from_meta(state)
    groups = FeatureGroup(state["feature_groups"]) if "feature_groups" in state else infer_feature_groups(in_features)

    extractor = ComputeHubFeatureExtractor(embedding=embedding_model, groups=groups)
    _emit(progress_cb, 15, "Extracting features")
    x, y = _collect_dataset(ann_paths, extractor)
    if len(x) == 0:
        raise ValueError("No labelled CALL nodes found in the annotation corpus.")
    if scaler is not None:
        x = scaler.transform(x)

    trainer = MLPTrainer(model=model, criterion=nn.CrossEntropyLoss(), learning_rate=lr)
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    final_loss = 0.0
    acc = 0.0
    for epoch in range(1, epochs + 1):
        final_loss, acc = trainer.train_epoch(x_t, y_t, epoch)
        _emit(progress_cb, 15 + int(80 * epoch / epochs), f"Epoch {epoch}/{epochs} loss={final_loss:.4f}")

    meta = {
        "in_features": in_features,
        "hidden": state.get("hidden", 256),
        "num_classes": state.get("num_classes", num_classes),
        "feature_groups": groups.value,
        "use_batchnorm": bool(model.use_batchnorm),
    }
    if scaler is not None:
        meta.update(scaler.to_meta())
    trainer.save_cpu_checkpoint(str(new_path), meta=meta)

    metrics = _metrics(len(ann_paths), x, y, float(acc), final_loss=float(final_loss))
    _emit(progress_cb, 100, f"Done (train acc {acc:.3f})")
    logger.info("[finetune] mlp checkpoint saved to %s", new_path)
    return new_path, metrics

def _finetune_hgt(
    base_ckpt: Path,
    ann_paths: list[Path],
    new_path: Path,
    *,
    epochs: int,
    lr: float,
    embedding_model: object | None,
    progress_cb: ProgressCB | None,
) -> tuple[Path, dict]:
    import json

    from torch_geometric.loader import DataLoader

    from SystemX.nn.data.v2.pruner import NullPruner
    from SystemX.nn.data.v2.tensor_encoder import HETERO_METADATA as _HETERO_METADATA
    from SystemX.nn.data.v2.tensor_encoder import EncoderV2
    from SystemX.nn.models.v2.gat import SystemXHeteroGraphTransformer

    num_classes = len(DOMAIN_EDGE_TYPES)
    state = torch.load(str(base_ckpt), map_location="cpu", weights_only=True)
    model_state = state.get("model_state_dict", state)

    if "feature_groups" in state:
        groups = FeatureGroup(state["feature_groups"])
    else:
        op_dim = int(model_state["lin_dict.operation.weight"].shape[1]) if "lin_dict.operation.weight" in model_state else 305
        groups = infer_feature_groups(op_dim)

    if embedding_model is None:
        from SystemX.nn.data.v2.fasttext_embedding import FastTextEmbeddingV2

        embedding_model = FastTextEmbeddingV2()

    encoder = EncoderV2(
        embedding_model=embedding_model,
        pruner=NullPruner(),
        feature_groups=groups,
        rich_variable_features=bool(state.get("rich_var_features", False)),
    )
    model = SystemXHeteroGraphTransformer(
        hidden_channels=int(state.get("hidden_channels", 128)),
        out_classes=num_classes,
        num_heads=int(state.get("num_heads", 4)),
        num_layers=int(state.get("num_layers", 3)),
        metadata=_HETERO_METADATA,
    )
    model.load_state_dict(model_state)

    _emit(progress_cb, 10, "Encoding annotation graphs")
    encoded: list = []
    for ann_path in ann_paths:
        try:
            with open(ann_path, encoding="utf-8") as f:
                raw = json.load(f)
            elements = raw if isinstance(raw, list) else raw.get("elements", [])
            nx_G = _annotation_to_networkx(elements)
            if nx_G.number_of_nodes() == 0:
                continue
            data = encoder.encode(nx_G)
            if data["operation"].train_mask.sum() == 0:
                continue
            encoded.append(data)
        except Exception as e:
            logger.warning("[finetune] skipping %s: %s", ann_path.name, e)

    if not encoded:
        raise ValueError("No trainable graphs after encoding the annotation corpus.")

    trainer = SystemXOnlineTrainer(model=model, criterion=nn.CrossEntropyLoss(ignore_index=-1), learning_rate=lr)
    final_loss = 0.0
    for epoch in range(1, epochs + 1):
        loader = DataLoader(encoded, batch_size=16, shuffle=True)
        epoch_loss, n_batches = 0.0, 0
        for batch in loader:
            epoch_loss += trainer.train_step(batch)
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)
        _emit(progress_cb, 10 + int(85 * epoch / epochs), f"Epoch {epoch}/{epochs} loss={final_loss:.4f}")

    trainer.save_cpu_checkpoint(
        str(new_path),
        meta={
            "hidden_channels": int(state.get("hidden_channels", 128)),
            "num_heads": int(state.get("num_heads", 4)),
            "num_layers": int(state.get("num_layers", 3)),
            "embedding": state.get("embedding", "fasttext"),
            "features": state.get("features"),
            "feature_groups": groups.value,
            "rich_var_features": bool(state.get("rich_var_features", False)),
        },
    )

    acc, n_nodes = _hgt_train_accuracy(model, encoded)
    metrics = {
        "train_accuracy": round(acc, 4),
        "final_loss": round(float(final_loss), 4),
        "num_annotations": len(ann_paths),
        "num_call_nodes": int(n_nodes),
    }
    _emit(progress_cb, 100, f"Done (train acc {acc:.3f})")
    logger.info("[finetune] hgt checkpoint saved to %s", new_path)
    return new_path, metrics

def _hgt_train_accuracy(model: object, encoded: list) -> tuple[float, int]:
    """Masked training accuracy over the encoded operation nodes."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data in encoded:
            logits = model(data.x_dict, data.edge_index_dict)
            mask = data["operation"].train_mask
            if mask.sum() == 0:
                continue
            preds = logits[mask].argmax(dim=1)
            targets = data["operation"].y[mask]
            correct += int((preds == targets).sum())
            total += int(mask.sum())
    return (correct / total if total else 0.0), total

def _metrics(n_ann: int, x: np.ndarray, y: np.ndarray, acc: float, final_loss: float | None) -> dict:
    m = {
        "train_accuracy": round(acc, 4),
        "num_annotations": n_ann,
        "num_call_nodes": int(len(x)),
        "label_distribution": {str(k): int(v) for k, v in Counter(y.astype(int).tolist()).items()},
    }
    if final_loss is not None:
        m["final_loss"] = round(final_loss, 4)
    return m
