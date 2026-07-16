import json
import logging
from pathlib import Path

import numpy as np

from SystemX.execution.trace import TraceSession, observed_order, parse_trace
from SystemX.nn.data.v2.execution_order_dataset import (
    CellPairFeaturizer,
    generate_candidate_pairs,
)
from SystemX.parser.graph_parser import GraphParser
from SystemX.sca.cell_graph_projector import CellGraphProjector

logger = logging.getLogger(__name__)

def is_valid_traced_session(session: TraceSession, cells: list[dict]) -> bool:
    """A session supervises when enough of its executed cells still exist in the snapshot to order (>= 4 distinct, matching the count-based minimum)."""
    cell_ids = {str(c.get("cell_id") or c.get("id")) for c in cells}
    executed_present = {e.cell_id for e in session.events if e.cell_id in cell_ids}
    return len(executed_present) >= 4

def _cells_as_executed(cells: list[dict], session: TraceSession, sources: dict[str, str]) -> list[dict]:
    """Snapshot cells with each executed cell's source swapped to the version that ran at its last execution, so features describe the executed code."""
    last_hash: dict[str, str] = {}
    for event in session.events:
        last_hash[event.cell_id] = event.source_hash

    effective = []
    for cell in cells:
        cell_id = str(cell.get("cell_id") or cell.get("id"))
        executed_source = sources.get(last_hash.get(cell_id, ""))
        if executed_source is not None and executed_source != cell.get("source"):
            cell = {**cell, "source": executed_source}
        effective.append(cell)
    return effective

def build_traced_notebook_pairs(
    cells: list[dict],
    session: TraceSession,
    featurizer: CellPairFeaturizer,
    parser: GraphParser | None = None,
    sources: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """(features, labels, is_nonlinear) for one traced session, else None."""
    if not is_valid_traced_session(session, cells):
        return None
    cells = _cells_as_executed(cells, session, sources or {})
    try:
        dfg = (parser or GraphParser()).parse(cells)
    except Exception as exc:
        logger.debug("Skipping traced session (parser failed: %s)", exc)
        return None

    cg = CellGraphProjector(dfg, cells).project()
    order = [c for c in observed_order(session) if c in cg.graph]
    last_rank = {cell: i for i, cell in enumerate(order)}
    dfg_graph = dfg.get_graph() if featurizer.hub_extractor else None

    xs, ys = [], []
    cell_cache: dict = {}
    for src, dst in generate_candidate_pairs(cg):
        if src not in last_rank or dst not in last_rank:
            continue
        label = 1.0 if last_rank[src] < last_rank[dst] else 0.0
        xs.append(featurizer.pair_vector(cg, src, dst, dfg_graph, cell_cache))
        ys.append(label)
        xs.append(featurizer.pair_vector(cg, dst, src, dfg_graph, cell_cache))
        ys.append(1.0 - label)

    if not xs:
        return None
    document_order = [c for c in cg.cells if c in last_rank]
    nonlinear = order != document_order
    return np.stack(xs), np.array(ys, dtype=np.float32), nonlinear

def build_traced_dataset(
    traces_dir: Path,
    featurizer: CellPairFeaturizer,
    nonlinear_weight: float = 2.0,
    limit: int | None = None,
) -> dict[str, np.ndarray]:
    """Scan *.trace.json sidecars under *traces_dir* into one pair dataset."""
    paths = sorted(traces_dir.glob("*.trace.json"))
    if limit:
        paths = paths[:limit]

    parser = GraphParser()
    all_x, all_y, all_w, all_nb = [], [], [], []
    names: list[str] = []
    stats = {"scanned": 0, "valid_sessions": 0, "nonlinear": 0}

    for path in paths:
        stats["scanned"] += 1
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skipping unreadable trace %s: %s", path, exc)
            continue
        sessions, sources = parse_trace(raw)
        cells = raw.get("cells_snapshot") or []
        if not isinstance(cells, list) or not cells:
            continue

        for session in sessions:
            built = build_traced_notebook_pairs(cells, session, featurizer, parser, sources)
            if built is None:
                continue
            x, y, nonlinear = built
            stats["valid_sessions"] += 1
            stats["nonlinear"] += int(nonlinear)
            nb_index = len(names)
            names.append(f"{path.name.removesuffix('.trace.json')}#{session.session_id}")
            all_x.append(x)
            all_y.append(y)
            all_w.append(np.full(len(y), nonlinear_weight if nonlinear else 1.0, dtype=np.float32))
            all_nb.append(np.full(len(y), nb_index, dtype=np.int32))

    logger.info(
        "Traced execution-order dataset: %d traces scanned, %d valid sessions (%d nonlinear), %d pairs.",
        stats["scanned"],
        stats["valid_sessions"],
        stats["nonlinear"],
        sum(len(y) for y in all_y),
    )
    if not all_x:
        return {"x": np.zeros((0, featurizer.pair_dim), dtype=np.float32), "y": np.zeros(0), "w": np.zeros(0), "notebook_id": np.zeros(0, dtype=np.int32), "notebooks": np.array([])}

    return {
        "x": np.concatenate(all_x).astype(np.float32),
        "y": np.concatenate(all_y),
        "w": np.concatenate(all_w),
        "notebook_id": np.concatenate(all_nb),
        "notebooks": np.array(names),
    }
