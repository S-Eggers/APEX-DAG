from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_LABELING_FAMILIES = ("hgt", "mlp", "xgboost")

def _family_of(manifest_key: str) -> str:
    return manifest_key.split("_", 1)[0]

def _is_finetune_key(manifest_key: str) -> bool:
    return "_ft" in manifest_key

class ModelRegistry:
    """Owns the shared models dict and the disk manifest for hot registration of variants."""

    def __init__(
        self,
        models: dict,
        paths: dict[str, Path],
        checkpoints_dir: str | Path | None,
        embedding_getter: Callable[[], object],
        load_fn: Callable[[str, Path, object, logging.Logger], object | None],
        logger: logging.Logger,
    ) -> None:
        self.models = models
        self.paths = dict(paths)
        self.checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir else None
        self._get_embedding = embedding_getter
        self._load_fn = load_fn
        self.log = logger
        self._lock = threading.Lock()

    def resolve(self, key: str) -> object | None:
        return self.models.get(key)

    def path_for(self, key: str) -> Path | None:
        return self.paths.get(key)

    def get_embedding(self) -> object:
        """Return the shared FastText embedding, loaded on first use."""
        return self._get_embedding()

    @property
    def manifest_path(self) -> Path | None:
        return (self.checkpoints_dir / "manifest.json") if self.checkpoints_dir else None

    @property
    def variants_path(self) -> Path | None:
        return (self.checkpoints_dir / "variants.json") if self.checkpoints_dir else None

    def register_variant(
        self,
        key: str,
        ckpt_path: str | Path,
        *,
        base_key: str,
        family: str,
        preset: str,
        metrics: dict,
        epochs: int,
        lr: float,
    ) -> None:
        """Load the checkpoint as a new variant, enable it, and persist the manifest and metadata."""
        ckpt_path = Path(ckpt_path)
        labeler = self._load_fn(key, ckpt_path, self._get_embedding(), self.log)
        if labeler is None:
            raise RuntimeError(f"Failed to load fine-tuned checkpoint for {key} from {ckpt_path}")

        created_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.models[key] = labeler
            self.paths[key] = ckpt_path
            self._append_manifest(key, ckpt_path)
            self._write_variant_meta(
                key,
                {
                    "base_key": base_key,
                    "family": family,
                    "preset": preset,
                    "created_at": created_at,
                    "epochs": epochs,
                    "lr": lr,
                    "metrics": metrics,
                },
            )
        self.log.info("SystemX: registered fine-tuned variant %s -> %s", key, ckpt_path.name)

    def list_variants(self) -> list[dict]:
        """Return metadata for every loaded labeling variant."""
        meta = self._read_variant_meta()
        out: list[dict] = []
        for key in sorted(self.models):
            family = _family_of(key)
            if family not in _LABELING_FAMILIES or key.startswith("exec_order"):
                continue
            info = meta.get(key)
            if info is not None:
                out.append(
                    {
                        "key": key,
                        "family": family,
                        "preset": info.get("preset"),
                        "base_key": info.get("base_key"),
                        "created_at": info.get("created_at"),
                        "metrics": info.get("metrics", {}),
                        "is_finetuned": True,
                        "loaded": True,
                    }
                )
            else:
                out.append(
                    {
                        "key": key,
                        "family": family,
                        "preset": key.split("_", 1)[1] if "_" in key else "",
                        "base_key": None,
                        "created_at": None,
                        "metrics": {},
                        "is_finetuned": _is_finetune_key(key),
                        "loaded": True,
                    }
                )
        return out

    def _append_manifest(self, key: str, ckpt_path: Path) -> None:
        mpath = self.manifest_path
        if mpath is None:
            self.log.warning("SystemX: no checkpoints_dir; cannot persist manifest for %s.", key)
            return
        manifest: dict[str, str] = {}
        if mpath.exists():
            try:
                manifest = json.loads(mpath.read_text())
            except (OSError, ValueError):
                self.log.warning("SystemX: manifest.json unreadable; recreating.", exc_info=True)
                manifest = {}
            mpath.with_suffix(".json.bak").write_text(json.dumps(manifest, indent=2))
        try:
            value = ckpt_path.name if ckpt_path.parent == mpath.parent else str(ckpt_path)
        except Exception:
            value = str(ckpt_path)
        manifest[key] = value
        mpath.write_text(json.dumps(manifest, indent=2))

    def _read_variant_meta(self) -> dict:
        vpath = self.variants_path
        if vpath is None or not vpath.exists():
            return {}
        try:
            return json.loads(vpath.read_text())
        except (OSError, ValueError):
            self.log.warning("SystemX: variants.json unreadable; ignoring.", exc_info=True)
            return {}

    def _write_variant_meta(self, key: str, info: dict) -> None:
        vpath = self.variants_path
        if vpath is None:
            return
        data = self._read_variant_meta()
        data[key] = info
        vpath.write_text(json.dumps(data, indent=2))
