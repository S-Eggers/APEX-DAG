#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

from sklearn.model_selection import KFold, StratifiedKFold

from SystemX.nn.training.v2.data_utils import annotation_to_networkx
from SystemX.sca.constants import COMPUTE_HUBS

logger = logging.getLogger(__name__)

def _graph_info(ann_path: Path) -> tuple[int, int]:
    """Return (n_call_nodes, proxy_label) for one annotation graph."""
    try:
        with open(ann_path, encoding="utf-8") as f:
            raw = json.load(f)
        elements = raw if isinstance(raw, list) else raw.get("elements", [])
        nx_G = annotation_to_networkx(elements)
    except Exception:
        return 0, -1

    n_call = sum(1 for _, a in nx_G.nodes(data=True) if int(a.get("node_type", -1)) in COMPUTE_HUBS)
    labels = [
        int(attrs.get("domain_label", -1))
        for _, attrs in nx_G.nodes(data=True)
        if int(attrs.get("node_type", -1)) in COMPUTE_HUBS and int(attrs.get("domain_label", -1)) >= 0
    ]
    proxy = Counter(labels).most_common(1)[0][0] if labels else -1
    return n_call, proxy

def make_folds(
    annotation_paths: list[Path],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[list[Path], list[Path]]]:
    """Build n_splits (train_paths, test_paths) folds over the annotation graphs."""
    all_paths = sorted(annotation_paths)

    info = {p: _graph_info(p) for p in all_paths}
    paths = [p for p in all_paths if info[p][0] > 0]
    dropped = len(all_paths) - len(paths)
    if dropped:
        logger.info("Dropped %d graph(s) with no CALL nodes; %d remain.", dropped, len(paths))

    n = len(paths)
    if n < n_splits:
        raise ValueError(f"Need at least n_splits={n_splits} graphs, got {n}.")

    proxy = [info[p][1] for p in paths]
    class_counts = Counter(proxy)
    stratifiable = all(c >= n_splits for c in class_counts.values())

    if stratifiable:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        index_iter = splitter.split(paths, proxy)
        logger.info("Stratified %d-fold CV over %d graphs. Class mix: %s", n_splits, n, dict(class_counts))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        index_iter = splitter.split(paths)
        logger.info("Plain %d-fold CV over %d graphs (a class had < %d graphs). Class mix: %s", n_splits, n, n_splits, dict(class_counts))

    folds: list[tuple[list[Path], list[Path]]] = []
    for train_idx, test_idx in index_iter:
        train_paths = [paths[i] for i in train_idx]
        test_paths = [paths[i] for i in test_idx]
        assert not (set(train_paths) & set(test_paths)), "train/test overlap"
        folds.append((train_paths, test_paths))

    tested = [p for _, test in folds for p in test]
    assert sorted(tested) == paths, "test sets do not partition the corpus"
    return folds
