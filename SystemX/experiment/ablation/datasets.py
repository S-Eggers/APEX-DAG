#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DATASETS: dict[str, tuple[str, str]] = {
    "jetbrains": ("data/jetbrains_dataset/annotations", "data/jetbrains_dataset/notebooks"),
    "github": ("data/github_dataset/annotations", "data/github_dataset/notebooks"),
    "jetbrains_cleaned": ("data/jetbrains_dataset_cleaned/annotations", "data/jetbrains_dataset/notebooks"),
    "github_cleaned": ("data/github_dataset_cleaned/annotations", "data/github_dataset/notebooks"),
    "combined_cleaned": ("data/combined_dataset_cleaned/annotations", "data/combined_dataset_cleaned/notebooks"),
}

COMBINED_SOURCES = ("jetbrains", "github")

GENERALIZATION_DATASETS = ("jetbrains", "github")

def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p

def build_combined_dataset(
    out_root: str | Path = "data/combined_dataset",
    sources: tuple[str, ...] = COMBINED_SOURCES,
) -> tuple[Path, Path]:
    """Materialise data/combined_dataset/{annotations,notebooks} as a symlink tree."""
    out_root = _abs(out_root)
    ann_out = out_root / "annotations"
    nb_out = out_root / "notebooks"
    ann_out.mkdir(parents=True, exist_ok=True)
    nb_out.mkdir(parents=True, exist_ok=True)

    linked = 0
    missing_nb = 0
    for src in sources:
        if src not in DATASETS:
            raise KeyError(f"Unknown source dataset: {src!r} (known: {sorted(DATASETS)})")
        ann_src = _abs(DATASETS[src][0])
        nb_src = _abs(DATASETS[src][1])
        for ann_path in sorted(ann_src.glob("*.json")):
            nb_path = nb_src / ann_path.with_suffix(".ipynb").name
            if not nb_path.exists():
                missing_nb += 1
                continue
            stem = f"{src}__{ann_path.stem}"
            _symlink(ann_path, ann_out / f"{stem}.json")
            _symlink(nb_path, nb_out / f"{stem}.ipynb")
            linked += 1

    logger.info(
        "Combined dataset -> %s  (%d pairs linked, %d annotations skipped for missing notebook)",
        out_root, linked, missing_nb,
    )
    return ann_out, nb_out

def _symlink(target: Path, link: Path) -> None:
    """Create/refresh link -> target (absolute target), idempotently."""
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() == target:
            return
        link.unlink()
    elif link.exists():
        link.unlink()
    link.symlink_to(target)

def resolve_dataset(name: str) -> tuple[Path, Path]:
    """Return absolute (annotations_dir, raw_dir) for a dataset name."""
    if name == "combined":
        return build_combined_dataset()
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset: {name!r} (known: {sorted(DATASETS) + ['combined']})")
    ann, raw = DATASETS[name]
    return _abs(ann), _abs(raw)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ann, raw = build_combined_dataset()
    print(f"annotations: {ann}\nnotebooks:   {raw}")
