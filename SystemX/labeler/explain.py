from __future__ import annotations

import logging

import numpy as np
from SystemX.nn.data.v2.feature_extractor import (
    _SCALAR_DESCRIPTIONS,
    ComputeHubFeatureExtractor,
)

logger = logging.getLogger(__name__)

_TOP_EMB_DIMS = 5

def group_importances(
    per_dim: np.ndarray,
    extractor: ComputeHubFeatureExtractor,
    predicted_class: int,
    *,
    model_name: str,
    values: np.ndarray | None = None,
) -> dict:
    """Aggregate a per-dimension attribution vector into named, normalized groups."""
    per_dim = np.abs(np.asarray(per_dim, dtype=np.float64))
    layout = extractor.feature_layout()
    total = float(per_dim.sum()) or 1.0

    vals = None if values is None else np.asarray(values, dtype=np.float64)

    groups: list[dict] = []
    for span in layout:
        seg = per_dim[span.start : span.end]
        seg_vals = None if vals is None else vals[span.start : span.end]
        seg_total = float(seg.sum()) or 1.0
        entry: dict = {
            "key": span.key,
            "name": span.name,
            "description": span.description,
            "score": float(seg.sum()) / total,
        }

        if span.scalar_names is None:
            order = np.argsort(seg)[::-1][:_TOP_EMB_DIMS]
            entry["dims"] = [
                {
                    "index": int(span.start + i),
                    "name": f"{span.key}[{int(i)}]",
                    "score": float(seg[i]) / seg_total,
                    **({} if seg_vals is None else {"value": float(seg_vals[i])}),
                }
                for i in order
                if seg[i] > 0
            ]
        else:
            entry["scalars"] = [
                {
                    "name": sname,
                    "description": _SCALAR_DESCRIPTIONS.get(sname, ""),
                    "score": float(seg[j]) / seg_total,
                    **({} if seg_vals is None else {"value": float(seg_vals[j])}),
                }
                for j, sname in enumerate(span.scalar_names)
            ]
        groups.append(entry)

    return {
        "model": model_name,
        "predicted_class": int(predicted_class),
        "groups": groups,
    }

def hgt_operation_importance(
    model: object,
    data: object,
    op_index_to_node_id: dict[int, object],
    extractor: ComputeHubFeatureExtractor,
    *,
    model_name: str = "HGT (input-gradient saliency)",
) -> dict[object, dict]:
    """Input-times-gradient attribution for every operation node under an HGT model."""
    import torch

    model.eval()
    x_dict = {k: v.detach() for k, v in data.x_dict.items()}
    x_op = x_dict["operation"].clone().requires_grad_(True)
    x_dict["operation"] = x_op

    logits = model(x_dict, data.edge_index_dict)
    preds = logits.argmax(dim=1)
    n_ops = logits.shape[0]

    out: dict[object, dict] = {}
    for op_idx, node_id in op_index_to_node_id.items():
        if op_idx >= n_ops:
            continue
        cls = int(preds[op_idx].item())
        grad = torch.autograd.grad(logits[op_idx, cls], x_op, retain_graph=True)[0]
        row_vals = x_op[op_idx].detach().cpu().numpy()
        sal = (x_op[op_idx] * grad[op_idx]).abs().detach().cpu().numpy()
        out[node_id] = group_importances(sal, extractor, cls, model_name=model_name, values=row_vals)
    return out

def xgboost_row_importance(
    model: object,
    x: np.ndarray,
    node_ids: list[object],
    extractor: ComputeHubFeatureExtractor,
    *,
    model_name: str = "XGBoost (SHAP)",
) -> dict[object, dict]:
    """Native per-instance feature contributions for an XGBoost classifier."""
    import xgboost as xgb

    x = np.asarray(x, dtype=np.float32)
    n_feat = extractor.feature_dim

    contribs = np.asarray(model.get_booster().predict(xgb.DMatrix(x), pred_contribs=True))
    preds = np.asarray(model.predict_proba(x)).argmax(axis=1)

    out: dict[object, dict] = {}
    for row, node_id in enumerate(node_ids):
        cls = int(preds[row])
        row_contrib = contribs[row, cls, :n_feat] if contribs.ndim == 3 else contribs[row, :n_feat]
        out[node_id] = group_importances(row_contrib, extractor, cls, model_name=model_name, values=x[row])
    return out
