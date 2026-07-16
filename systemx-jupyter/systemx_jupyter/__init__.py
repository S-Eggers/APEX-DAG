import json
import logging
from pathlib import Path
from typing import Callable

try:
    from ._version import __version__
except ImportError:
    import warnings

    warnings.warn("Importing 'systemx_jupyter' outside a proper installation.", stacklevel=2)
    __version__ = "dev"

from SystemX.util.logger import configure_systemx_logger

from .config import SystemXConfig
from .handlers import setup_handlers

_MODEL_FAMILIES = ("hgt", "mlp", "xgboost")
_FEATURE_PRESETS = ("standard", "all", "emb_only", "api_lib", "struct_only")
_EXEC_ORDER_KEYS = tuple(f"exec_order_{fam}_{preset}" for fam in ("mlp", "xgboost") for preset in ("struct", "standard"))
_MANIFEST_KEYS: list[str] = [f"{fam}_{preset}" for fam in _MODEL_FAMILIES for preset in _FEATURE_PRESETS] + list(_EXEC_ORDER_KEYS)

_GLOB_SUFFIX = {"hgt": ".pt", "mlp": ".pt", "xgboost": ".json"}

def _jupyter_labextension_paths() -> list[dict]:
    return [{"src": "labextension", "dest": "systemx-jupyter"}]

def _jupyter_server_extension_points() -> list[dict]:
    return [{"module": "systemx_jupyter"}]

def _family_of(manifest_key: str) -> str:
    """Return the model family prefix of a manifest key."""
    return manifest_key.split("_", 1)[0]

def _load_one_labeler(manifest_key: str, path: Path, embedding: object, logger: logging.Logger) -> object | None:
    """Load a single learned labeler for the given manifest key from its checkpoint."""
    family = _family_of(manifest_key)
    try:
        if family == "hgt":
            from SystemX.labeler.hgt_labeler import HGTLabeler

            labeler = HGTLabeler.from_checkpoint(str(path), embedding_model=embedding)
        elif family == "mlp":
            from SystemX.labeler.mlp_labeler import MLPLabeler

            labeler = MLPLabeler.from_checkpoint(str(path), embedding_model=embedding)
        elif family == "xgboost":
            from SystemX.labeler.xgboost_labeler import XGBoostLabeler

            labeler = XGBoostLabeler.from_checkpoint(str(path), embedding_model=embedding)
        else:
            logger.warning("SystemX: Unknown model family for manifest key %r; skipping.", manifest_key)
            return None
        logger.info("SystemX: Loaded %s from %s", manifest_key, path.name)
        return labeler
    except Exception as exc:
        logger.error("SystemX: Failed loading %s (%s): %s", manifest_key, path.name, exc, exc_info=True)
        return None

def _resolve_checkpoint_paths(checkpoints_dir: str, logger: logging.Logger) -> dict[str, Path]:
    """Map each manifest key to its checkpoint path, from manifest.json or a glob fallback."""
    resolved: dict[str, Path] = {}
    if not checkpoints_dir:
        return resolved

    ckpt_dir = Path(checkpoints_dir)
    if not ckpt_dir.is_dir():
        logger.warning("SystemX: v2_checkpoints_dir does not exist: %s", checkpoints_dir)
        return resolved

    manifest_path = ckpt_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest: dict[str, str] = json.load(f)
        except (OSError, ValueError) as exc:
            logger.error("SystemX: Failed to read %s: %s", manifest_path, exc)
            manifest = {}
        for key in sorted(set(manifest) | set(_MANIFEST_KEYS)):
            raw = manifest.get(key)
            if not raw:
                continue
            p = Path(raw)
            if not p.is_absolute():
                candidate = ckpt_dir / raw
                p = candidate if candidate.exists() else p
            if p.exists():
                resolved[key] = p
            else:
                logger.warning("SystemX: manifest %s -> %s not found on disk.", key, raw)
        logger.info("SystemX: manifest.json resolved %d/%d backend variants.", len(resolved), len(_MANIFEST_KEYS))
        return resolved

    logger.info("SystemX: no manifest.json in %s; falling back to glob discovery.", ckpt_dir)
    for key in _MANIFEST_KEYS:
        suffix = _GLOB_SUFFIX["mlp" if "mlp" in key else "xgboost"] if key in _EXEC_ORDER_KEYS else _GLOB_SUFFIX[_family_of(key)]
        matches = sorted(p for p in ckpt_dir.glob(f"{key}*{suffix}") if not p.name.endswith(".meta.json"))
        if matches:
            resolved[key] = matches[-1]
    return resolved

def make_embedding_getter(logger: logging.Logger) -> "Callable[[], object]":
    """Return a lazy, memoized getter for the shared FastText embedding."""
    _cache: dict[str, object] = {}

    def get_embedding() -> object:
        if "emb" not in _cache:
            from SystemX.nn.data.v2.fasttext_embedding import FastTextEmbeddingV2

            logger.info("SystemX: loading shared FastText embedding.")
            _cache["emb"] = FastTextEmbeddingV2()
        return _cache["emb"]

    return get_embedding

def _discover_v2_models(
    checkpoints_dir: str,
    logger: logging.Logger,
    embedding_getter: "Callable[[], object] | None" = None,
) -> tuple[dict[str, object], dict[str, Path]]:
    """Load every checkpoint found under the checkpoints directory."""
    resolved = _resolve_checkpoint_paths(checkpoints_dir, logger)
    if not resolved:
        return {}, {}

    get_embedding = embedding_getter or make_embedding_getter(logger)

    result: dict[str, object] = {}
    for key, path in resolved.items():
        if key in _EXEC_ORDER_KEYS:
            from SystemX.execution.learned_predictor import load_exec_order_predictor

            predictor = load_exec_order_predictor(key, path, get_embedding)
            if predictor is not None:
                logger.info("SystemX: Loaded %s from %s", key, path.name)
                result[key] = predictor
            continue
        labeler = _load_one_labeler(key, path, get_embedding(), logger)
        if labeler is not None:
            result[key] = labeler
    loaded_paths = {k: resolved[k] for k in result}
    return result, loaded_paths

def _load_jupyter_server_extension(server_app: object) -> None:
    configure_systemx_logger(server_app.log)
    server_app.log.info("SystemX: Initialising server extension.")

    systemx_config = SystemXConfig(config=server_app.config)

    ckpt_dir = systemx_config.v2_checkpoints_dir
    if not ckpt_dir:
        fallback = Path.cwd() / "checkpoints" / "v2"
        if fallback.is_dir():
            ckpt_dir = str(fallback)
            server_app.log.info("SystemX: Auto-detected v2_checkpoints_dir: %s", fallback)

    get_embedding = make_embedding_getter(server_app.log)
    models, paths = _discover_v2_models(ckpt_dir, server_app.log, get_embedding)

    server_app.log.info("SystemX: Loaded backends: %s", sorted(models))

    from .model_registry import ModelRegistry

    registry = ModelRegistry(
        models=models,
        paths=paths,
        checkpoints_dir=ckpt_dir,
        embedding_getter=get_embedding,
        load_fn=_load_one_labeler,
        logger=server_app.log,
    )

    setup_handlers(server_app.web_app, models, server_app.config, registry=registry)
    server_app.log.info("SystemX: Handlers registered.")
