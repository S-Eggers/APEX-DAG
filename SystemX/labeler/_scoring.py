from __future__ import annotations

import numpy as np
from SystemX.sca.constants import canonical_domain_label

def top2_and_confidence(prob_row: np.ndarray) -> tuple[int, int, float, float]:
    """Return (top_label, runner_up_label, top_probability, margin), labels canonicalized."""
    order = np.argsort(prob_row)[::-1]
    top = canonical_domain_label(int(order[0]))
    top_p = float(prob_row[order[0]])
    if order.size > 1:
        runner = canonical_domain_label(int(order[1]))
        margin = top_p - float(prob_row[order[1]])
    else:
        runner = top
        margin = 1.0
    return top, runner, top_p, margin

def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax for a [n, c] logit matrix."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
