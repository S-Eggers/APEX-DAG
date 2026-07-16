#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_VENV_PYTHON = _HERE.parent.parent.parent / ".venv" / "bin" / "python3"
_PYTHON = str(_VENV_PYTHON) if _VENV_PYTHON.exists() else sys.executable

HGT_VARIANTS = [
    ("hgt_standard", "standard"),
    ("hgt_all", "all"),
    ("hgt_emb_only", "emb_only"),
    ("hgt_api_lib", "api_lib"),
    ("hgt_api29", "api29"),
    ("hgt_api24", "api24"),
    ("hgt_struct_only", "struct_only"),
]

BALANCING_BASELINE_VARIANTS = [
    ("mlp_undersample", "mlp", "undersample"),
    ("mlp_classweight", "mlp", "class_weight"),
    ("xgboost_undersample", "xgboost", "undersample"),
    ("xgboost_classweight", "xgboost", "class_weight"),
]
BALANCING_HGT_VARIANTS = [
    ("hgt_cw_none", "none"),
    ("hgt_cw_balanced", "balanced"),
]

MLP_VARIANTS = [
    ("mlp_standard", "standard"),
    ("mlp_all", "all"),
    ("mlp_emb_only", "emb_only"),
    ("mlp_api_lib", "api_lib"),
    ("mlp_api29", "api29"),
    ("mlp_api24", "api24"),
    ("mlp_struct_only", "struct_only"),
]

XGBOOST_VARIANTS = [
    ("xgboost_standard", "standard"),
    ("xgboost_all", "all"),
    ("xgboost_emb_only", "emb_only"),
    ("xgboost_api_lib", "api_lib"),
    ("xgboost_api29", "api29"),
    ("xgboost_api24", "api24"),
    ("xgboost_struct_only", "struct_only"),
    ("xgboost_emb_rich", "emb_rich"),
]

def _find_latest(directory: Path, glob: str) -> Path | None:
    files = sorted(directory.glob(glob))
    return files[-1] if files else None

def _run(cmd: list[str], allow_failure: bool = False) -> bool:
    """Run a command; returns True on success, False on failure (if allow_failure)."""
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logger.error("Command failed (exit %d): %s", result.returncode, " ".join(cmd))
        if allow_failure:
            return False
        sys.exit(result.returncode)
    return True

def _train_list_args(train_list: Path | None) -> list[str]:
    return ["--train_list", str(train_list)] if train_list is not None else []

def train_hgt(
    features: str,
    annotations_dir: Path,
    output_dir: Path,
    epochs: int,
    manifest_key: str,
    train_list: Path | None = None,
    embedding: str = "fasttext",
    allow_failure: bool = False,
    class_weighting: str | None = None,
) -> Path | None:
    variant_dir = output_dir / manifest_key
    variant_dir.mkdir(parents=True, exist_ok=True)
    ok = _run(
        [
            _PYTHON,
            "-m",
            "SystemX.nn.training.v2.train_hgt",
            "--annotations_dir",
            str(annotations_dir),
            "--output_dir",
            str(variant_dir),
            "--embedding",
            embedding,
            "--features",
            features,
            "--epochs",
            str(epochs),
            *(["--class_weighting", class_weighting] if class_weighting is not None else []),
            *_train_list_args(train_list),
        ],
        allow_failure=allow_failure,
    )
    if not ok:
        return None
    ckpt = _find_latest(variant_dir, "hgt_*.pt")
    if ckpt is None:
        raise RuntimeError(f"No checkpoint found after training {manifest_key}")
    return ckpt

def train_mlp(
    features: str,
    annotations_dir: Path,
    output_dir: Path,
    epochs: int,
    manifest_key: str,
    train_list: Path | None = None,
    sampling: str = "none",
) -> Path | None:
    variant_dir = output_dir / manifest_key
    variant_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _PYTHON,
            "-m",
            "SystemX.nn.training.v2.train_baselines",
            "--annotations_dir",
            str(annotations_dir),
            "--output_dir",
            str(variant_dir),
            "--model",
            "mlp",
            "--features",
            features,
            "--epochs",
            str(epochs),
            "--sampling",
            sampling,
            *_train_list_args(train_list),
        ]
    )
    ckpt = _find_latest(variant_dir, "mlp_baseline_*.pt")
    if ckpt is None:
        raise RuntimeError(f"No checkpoint found after training {manifest_key}")
    return ckpt

def train_xgboost(
    features: str,
    annotations_dir: Path,
    output_dir: Path,
    manifest_key: str,
    train_list: Path | None = None,
    sampling: str = "none",
) -> Path:
    variant_dir = output_dir / manifest_key
    variant_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _PYTHON,
            "-m",
            "SystemX.nn.training.v2.train_baselines",
            "--annotations_dir",
            str(annotations_dir),
            "--output_dir",
            str(variant_dir),
            "--model",
            "xgboost",
            "--features",
            features,
            "--sampling",
            sampling,
            *_train_list_args(train_list),
        ]
    )
    ckpt = _find_latest(variant_dir, "xgboost_baseline_*.json")
    if ckpt is None:
        raise RuntimeError(f"No checkpoint found after training {manifest_key}")
    return ckpt

def main() -> None:
    parser = argparse.ArgumentParser(description="Train all SystemX model variants")
    parser.add_argument("--annotations_dir", default="data/jetbrains_dataset/annotations")
    parser.add_argument("--output_dir", default="output/checkpoints")
    parser.add_argument("--epochs_hgt", type=int, default=80, help="Epochs for HGT+FastText (80 with mini-batching)")
    parser.add_argument("--epochs_mlp", type=int, default=50)
    parser.add_argument("--train_list", default=None, help="Optional file listing the fold's training annotation filenames (one per line). Restricts every model to this train split.")
    parser.add_argument("--features", default=None, help="Comma-separated HGT presets to train (e.g. 'standard'); default all 5. 'standard' = fast headline-only run.")
    parser.add_argument("--balancing", action="store_true", help="Also train the class-imbalance ablation variants (under-sampling / class-weighting, STANDARD features).")
    parser.add_argument("--force", action="store_true", help="Retrain even if checkpoint exists")
    args = parser.parse_args()

    wanted = {f.strip() for f in args.features.split(",")} if args.features else None

    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_list = Path(args.train_list) if args.train_list else None

    manifest_path = output_dir / "manifest.json"
    manifest: dict[str, str] = {}

    if manifest_path.exists() and not args.force:
        with open(manifest_path) as f:
            manifest = json.load(f)
        logger.info("Loaded existing manifest with %d entries.", len(manifest))

    def _need_train(key: str) -> bool:
        if args.force:
            return True
        if key not in manifest:
            return True
        return not Path(manifest[key]).exists()

    for key, features in HGT_VARIANTS:
        if wanted is not None and features not in wanted:
            logger.info("Skipping HGT [%s] - not in --features.", features)
            continue
        if _need_train(key):
            logger.info("Training HGT [%s] (epochs=%d) ...", features, args.epochs_hgt)
            ckpt = train_hgt(features, annotations_dir, output_dir, args.epochs_hgt, key, train_list=train_list, allow_failure=True)
            if ckpt is None:
                logger.warning("HGT [%s] training failed - skipping.", features)
                continue
            manifest[key] = str(ckpt)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info("  -> saved to %s", ckpt)
        else:
            logger.info("Skipping HGT [%s] - checkpoint exists.", features)

    for key, features in MLP_VARIANTS:
        if _need_train(key):
            logger.info("Training MLP [%s] ...", features)
            ckpt = train_mlp(features, annotations_dir, output_dir, args.epochs_mlp, key, train_list=train_list)
            manifest[key] = str(ckpt)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info("  -> saved to %s", ckpt)
        else:
            logger.info("Skipping MLP [%s] - checkpoint exists.", features)

    for key, features in XGBOOST_VARIANTS:
        if _need_train(key):
            logger.info("Training XGBoost [%s] ...", features)
            ckpt = train_xgboost(features, annotations_dir, output_dir, key, train_list=train_list)
            manifest[key] = str(ckpt)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info("  -> saved to %s", ckpt)
        else:
            logger.info("Skipping XGBoost [%s] - checkpoint exists.", features)

    if args.balancing:
        for key, model, sampling in BALANCING_BASELINE_VARIANTS:
            if not _need_train(key):
                logger.info("Skipping %s - checkpoint exists.", key)
                continue
            logger.info("Training %s [%s sampling=%s] ...", key, model, sampling)
            if model == "mlp":
                ckpt = train_mlp("standard", annotations_dir, output_dir, args.epochs_mlp, key, train_list=train_list, sampling=sampling)
            else:
                ckpt = train_xgboost("standard", annotations_dir, output_dir, key, train_list=train_list, sampling=sampling)
            manifest[key] = str(ckpt)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info("  -> saved to %s", ckpt)

        for key, class_weighting in BALANCING_HGT_VARIANTS:
            if not _need_train(key):
                logger.info("Skipping %s - checkpoint exists.", key)
                continue
            logger.info("Training %s [HGT class_weighting=%s] ...", key, class_weighting)
            ckpt = train_hgt("standard", annotations_dir, output_dir, args.epochs_hgt, key, train_list=train_list, class_weighting=class_weighting, allow_failure=True)
            if ckpt is None:
                logger.warning("HGT [%s] training failed - skipping.", key)
                continue
            manifest[key] = str(ckpt)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info("  -> saved to %s", ckpt)

    logger.info("All training complete. Manifest: %s", manifest_path)
    for k, v in manifest.items():
        logger.info("  %s -> %s", k, v)

if __name__ == "__main__":
    main()
