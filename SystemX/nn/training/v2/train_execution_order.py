import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from SystemX.nn.data.v2.execution_order_dataset import (
    CellPairFeaturizer,
    build_dataset,
    build_selfsupervised_dataset,
    save_dataset,
)
from SystemX.nn.models.v2.mlp import ComputeHubMLP

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("SystemX.nn.training.v2.train_execution_order")

def _make_featurizer(preset: str):
    if preset == "struct":
        return CellPairFeaturizer(None)
    from SystemX.nn.data.v2.fasttext_embedding import FastTextEmbeddingV2
    from SystemX.nn.data.v2.feature_extractor import ComputeHubFeatureExtractor, FeatureGroup

    groups = FeatureGroup.STANDARD & ~FeatureGroup.CELL_POS
    return CellPairFeaturizer(ComputeHubFeatureExtractor(FastTextEmbeddingV2(), groups=groups))

def _run_epochs(model, optimizer, criterion, x: np.ndarray, y: np.ndarray, w: np.ndarray, epochs: int, tag: str) -> None:
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    w_t = torch.tensor(w, dtype=torch.float32)
    w_t = w_t / w_t.mean()

    log_every = max(1, epochs // 5)
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        logits = model(x_t)
        loss = (criterion(logits, y_t) * w_t).mean()
        loss.backward()
        optimizer.step()
        if epoch % log_every == 0 or epoch == epochs:
            acc = (logits.argmax(dim=1) == y_t).float().mean().item()
            logger.info("MLP %s epoch %d/%d  loss=%.4f  acc=%.3f", tag, epoch, epochs, loss.item(), acc)

def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    epochs: int,
    output_dir: Path,
    preset: str,
    pretrain: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    pretrain_epochs: int = 0,
) -> Path:
    model = ComputeHubMLP(in_features=x.shape[1], num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(reduction="none")

    if pretrain is not None and pretrain_epochs > 0:
        px, py, pw = pretrain
        logger.info("Pretraining on %d permutation-restoration pairs for %d epochs.", len(py), pretrain_epochs)
        _run_epochs(model, optimizer, criterion, px, py, pw, pretrain_epochs, "pretrain")
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    _run_epochs(model, optimizer, criterion, x, y, w, epochs, "train")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"exec_order_mlp_{preset}_{timestamp}.pt"
    model.eval()
    torch.save(
        {
            "model_state_dict": model.to("cpu").state_dict(),
            "meta": {
                "task": "exec_order",
                "in_features": x.shape[1],
                "hidden": 256,
                "num_classes": 2,
                "featurizer_preset": preset,
                "pretrain_epochs": pretrain_epochs if pretrain is not None else 0,
            },
        },
        path,
    )
    logger.info("Exec-order MLP checkpoint saved to %s", path)
    return path

def train_xgboost(x: np.ndarray, y: np.ndarray, w: np.ndarray, output_dir: Path, preset: str) -> Path:
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(x, y.astype(int), sample_weight=w)
    acc = (model.predict(x) == y.astype(int)).mean()
    logger.info("Exec-order XGBoost training accuracy: %.3f", acc)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"exec_order_xgboost_{preset}_{timestamp}.json"
    model.save_model(str(path))
    path.with_suffix(".meta.json").write_text(json.dumps({"task": "exec_order", "featurizer_preset": preset, "in_features": int(x.shape[1])}))
    logger.info("Exec-order XGBoost model saved to %s", path)
    return path

def update_manifest(output_dir: Path, entries: dict[str, Path]) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, str] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except ValueError:
            logger.warning("Existing manifest.json unreadable; recreating with exec-order entries only.")
    for key, path in entries.items():
        manifest[key] = path.name
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("manifest.json updated with %s", sorted(entries))

def main() -> None:
    parser = argparse.ArgumentParser(description="Train execution-order (pairwise cell precedence) models")
    parser.add_argument("--notebooks_dir", help="Directory of .ipynb files with execution counts (e.g. data/jetbrains_dataset/notebooks)")
    parser.add_argument("--dataset", default=None, help="Load a previously saved .npz pair dataset instead of building one")
    parser.add_argument("--save_dataset", default=None, help="Optional path to cache the built pair dataset (.npz)")
    parser.add_argument("--output_dir", default="./checkpoints/v2")
    parser.add_argument("--model", default="both", choices=["mlp", "xgboost", "both"])
    parser.add_argument("--epochs", type=int, default=100, help="MLP epochs")
    parser.add_argument("--features", default="struct", choices=["struct", "standard"], help="struct = cell-graph scalars only; standard = + pooled CALL-node embeddings (needs FastText)")
    parser.add_argument("--nonlinear_weight", type=float, default=2.0, help="Sample weight for pairs from nonlinear (out-of-document-order) notebooks (~34%% of valid notebooks; 2x puts them at rough parity)")
    parser.add_argument("--limit", type=int, default=None, help="Max notebooks to scan")
    parser.add_argument("--train_list", default=None, help="Optional file listing the fold's training notebook filenames (one per line)")
    parser.add_argument("--no_manifest", action="store_true", help="Do not update manifest.json")
    parser.add_argument(
        "--augment_permutations",
        type=int,
        default=0,
        help="Self-supervision: permutations per eligible notebook whose restoration pairs are added to training (needs no execution counts)",
    )
    parser.add_argument("--permutation_weight", type=float, default=1.0, help="Sample weight for permutation-restoration pairs when mixed into training")
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=0,
        help="MLP only: pretrain on the permutation-restoration pairs for N epochs, then fine-tune on the supervised pairs (XGBoost always trains on the mixed set)",
    )
    parser.add_argument("--random_fraction", type=float, default=0.3, help="Fraction of unconstrained shuffles among the permutations (rest are linear extensions)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_permutations = args.augment_permutations > 0 or args.pretrain_epochs > 0
    names = None
    if args.train_list and Path(args.train_list).exists():
        names = [ln.strip() for ln in Path(args.train_list).read_text().splitlines() if ln.strip()]

    featurizer = None
    if not args.dataset or use_permutations:
        featurizer = _make_featurizer(args.features)

    if args.dataset:
        data = dict(np.load(args.dataset, allow_pickle=True))
        logger.info("Loaded cached pair dataset: %d pairs.", len(data["y"]))
    else:
        if not args.notebooks_dir:
            parser.error("--notebooks_dir is required when no --dataset is given")
        data = build_dataset(
            Path(args.notebooks_dir),
            featurizer,
            nonlinear_weight=args.nonlinear_weight,
            limit=args.limit,
            notebook_names=names,
        )
        if args.save_dataset:
            save_dataset(data, Path(args.save_dataset))

    x, y, w = data["x"], data["y"], data["w"]
    if len(y) == 0:
        logger.error("No training pairs produced - check that the notebooks carry valid execution counts.")
        return
    logger.info("Training on %d supervised pairs (%d features), positive rate %.3f.", len(y), x.shape[1], float(y.mean()))

    pretrain = None
    if use_permutations:
        if not args.notebooks_dir:
            parser.error("--notebooks_dir is required for permutation augmentation")
        perm_data = build_selfsupervised_dataset(
            Path(args.notebooks_dir),
            featurizer,
            n_permutations=args.augment_permutations or 3,
            random_fraction=args.random_fraction,
            limit=args.limit,
            notebook_names=names,
        )
        px, py = perm_data["x"], perm_data["y"]
        pw = np.full(len(py), args.permutation_weight, dtype=np.float32)
        if len(py) == 0:
            logger.warning("No permutation-restoration pairs produced; continuing without self-supervision.")
        elif args.pretrain_epochs > 0:
            pretrain = (px, py, pw)
            logger.info("Pretrain corpus: %d permutation pairs.", len(py))
        else:
            x = np.concatenate([x, px])
            y = np.concatenate([y, py])
            w = np.concatenate([w, pw])
            logger.info("Mixed in %d permutation pairs (weight %.1f) -> %d total.", len(py), args.permutation_weight, len(y))

    entries: dict[str, Path] = {}
    if args.model in ("mlp", "both"):
        entries[f"exec_order_mlp_{args.features}"] = train_mlp(x, y, w, args.epochs, output_dir, args.features, pretrain=pretrain, pretrain_epochs=args.pretrain_epochs)
    if args.model in ("xgboost", "both"):
        if pretrain is not None:
            xx, yy, ww = np.concatenate([x, pretrain[0]]), np.concatenate([y, pretrain[1]]), np.concatenate([w, pretrain[2]])
        else:
            xx, yy, ww = x, y, w
        entries[f"exec_order_xgboost_{args.features}"] = train_xgboost(xx, yy, ww, output_dir, args.features)

    if entries and not args.no_manifest:
        update_manifest(output_dir, entries)

if __name__ == "__main__":
    main()
