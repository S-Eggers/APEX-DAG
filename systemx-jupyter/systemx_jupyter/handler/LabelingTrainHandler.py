from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import tornado
from jupyter_server.base.handlers import APIHandler

from ..policy.WorkspacePolicy import WorkspacePolicy

_TRAINABLE_FAMILIES = ("hgt", "mlp", "xgboost")

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="systemx-finetune")
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_ACTIVE: set[str] = set()

_MAX_EPOCHS = 50
_DEFAULT_EPOCHS = 5
_DEFAULT_LR = 5e-4

def _set(job_id: str, **fields: object) -> None:
    with _JOBS_LOCK:
        job = _JOBS.setdefault(job_id, {})
        job.update(fields)

def _get(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job is not None else None

def _run_job(job_id: str, params: dict, registry: object, log: object) -> None:
    try:
        _set(job_id, status="running", progress=0, message="Starting")

        def cb(pct: int, msg: str) -> None:
            _set(job_id, progress=int(pct), message=msg)

        from SystemX.nn.training.v2.finetune import finetune_variant

        new_path, metrics = finetune_variant(
            base_ckpt=params["base_ckpt"],
            family=params["family"],
            annotations_dir=params["annotations_dir"],
            output_dir=params["output_dir"],
            out_stem=params["new_key"],
            epochs=params["epochs"],
            lr=params["lr"],
            progress_cb=cb,
            embedding_model=registry.get_embedding(),
        )
        registry.register_variant(
            params["new_key"],
            new_path,
            base_key=params["base_key"],
            family=params["family"],
            preset=params["preset"],
            metrics=metrics,
            epochs=params["epochs"],
            lr=params["lr"],
        )
        _set(
            job_id,
            status="done",
            progress=100,
            message=f"Registered {params['new_key']}",
            variant=params["new_key"],
            metrics=metrics,
        )
    except Exception as e:  # noqa: BLE001 - surface any failure to the poller
        log.error("SystemX: training job %s failed: %s", job_id, e, exc_info=True)
        _set(job_id, status="error", message=str(e))
    finally:
        with _JOBS_LOCK:
            _ACTIVE.discard(job_id)

class LabelingTrainHandler(APIHandler):
    def initialize(self, registry: object = None, jupyter_server_app_config: dict | None = None) -> None:
        self.registry = registry
        self.jupyter_server_app_config = jupyter_server_app_config

    @tornado.web.authenticated
    def get(self) -> None:
        """Return the status of a fine-tune job by its job_id."""
        job_id = self.get_argument("job_id", "")
        job = _get(job_id)
        if job is None:
            self.set_status(404)
            self.finish(json.dumps({"success": False, "message": f"Unknown job_id {job_id!r}."}))
            return
        self.finish(json.dumps({"success": True, "job_id": job_id, **job}))

    @tornado.web.authenticated
    def post(self) -> None:
        """Start a fine-tune job and return its job_id."""
        if self.registry is None or self.registry.checkpoints_dir is None:
            self.set_status(200)
            self.finish(
                json.dumps(
                    {
                        "success": False,
                        "message": "No checkpoints directory is configured; cannot persist a new variant.",
                    }
                )
            )
            return

        body: dict = self.get_json_body() or {}
        variant: str = (body.get("modelVariant") or "").strip()
        backend: str = body.get("nnBackend", "hgt")
        preset_in: str = body.get("featurePreset", "standard")
        base_path: str = body.get("base_path", "")

        base_key = variant if variant else f"{backend}_{preset_in}"
        root = base_key.split("_ft")[0]
        family = root.split("_", 1)[0]
        preset = root.split("_", 1)[1] if "_" in root else ""

        if family not in _TRAINABLE_FAMILIES:
            self._fail(f"Model family {family!r} cannot be fine-tuned (trainable: {_TRAINABLE_FAMILIES}).")
            return

        base_ckpt = self.registry.path_for(base_key)
        if base_ckpt is None:
            self._fail(f"Base model {base_key!r} is not loaded; cannot fine-tune from it.")
            return

        if not base_path:
            self._fail("base_path (the dataset directory) is required to locate saved annotations.")
            return
        try:
            annotations_dir = WorkspacePolicy(Path.cwd(), base_path).annotations_dir
        except (ValueError, PermissionError) as e:
            self._fail(f"Invalid base_path: {e}")
            return
        if not annotations_dir.is_dir() or not any(annotations_dir.glob("*.json")):
            self._fail(f"No saved annotations found in {annotations_dir}. Save some annotations first.")
            return

        with _JOBS_LOCK:
            if _ACTIVE:
                running = next(iter(_ACTIVE))
                self.set_status(200)
                self.finish(
                    json.dumps(
                        {
                            "success": False,
                            "message": "A training job is already running.",
                            "job_id": running,
                        }
                    )
                )
                return
            job_id = uuid.uuid4().hex
            _ACTIVE.add(job_id)

        epochs = max(1, min(int(body.get("epochs", _DEFAULT_EPOCHS)), _MAX_EPOCHS))
        lr = float(body.get("lr", _DEFAULT_LR))
        new_key = f"{root}_ft{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        params = {
            "base_ckpt": base_ckpt,
            "base_key": base_key,
            "family": family,
            "preset": preset,
            "new_key": new_key,
            "annotations_dir": annotations_dir,
            "output_dir": self.registry.checkpoints_dir,
            "epochs": epochs,
            "lr": lr,
        }
        _set(job_id, status="queued", progress=0, message="Queued", variant=None)
        _EXECUTOR.submit(_run_job, job_id, params, self.registry, self.log)

        self.log.info("SystemX: started fine-tune job %s (base=%s -> %s)", job_id, base_key, new_key)
        self.finish(json.dumps({"success": True, "job_id": job_id, "variant": new_key}))

    def _fail(self, message: str) -> None:
        self.set_status(200)
        self.finish(json.dumps({"success": False, "message": message}))

    def data_received(self, chunk: bytes) -> None:
        pass
