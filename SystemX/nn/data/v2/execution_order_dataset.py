import json
import logging
import random
from pathlib import Path

import numpy as np

from SystemX.parser.graph_parser import GraphParser
from SystemX.sca.cell_graph_projector import CellDependencyGraph, CellGraphProjector, DependencyKind
from SystemX.sca.constants import NODE_TYPES

logger = logging.getLogger(__name__)

_KIND_ORDER = [k.name for k in DependencyKind]

_CELL_SCALARS = [
    "n_defs",
    "n_uses",
    "n_undefined",
    "has_opaque",
    "parse_error",
    "log_n_lines",
    "in_degree",
    "out_degree",
]

def is_valid_supervision(cells: list[dict], strict: bool = False) -> bool:
    """Counts must yield trustworthy pairwise precedence labels."""
    counts = [c.get("execution_count") for c in cells]
    if len(cells) < 4:
        return False
    if any(not isinstance(c, int) or c <= 0 for c in counts):
        return False
    if len(set(counts)) != len(counts):
        return False
    if strict and max(counts) != len(counts):
        return False
    return True

def is_linear(cells: list[dict]) -> bool:
    counts = [c.get("execution_count") for c in cells]
    return counts == sorted(counts)

class CellPairFeaturizer:
    """Feature vectors for (cell_i, cell_j) pairs over a CellDependencyGraph."""

    def __init__(self, hub_extractor=None) -> None:
        if hub_extractor is not None:
            from SystemX.nn.data.v2.feature_extractor import FeatureGroup

            if FeatureGroup.CELL_POS in hub_extractor.groups:
                raise ValueError("hub_extractor must exclude CELL_POS - document position may only enter as the explicit doc_delta pair feature")
        self.hub_extractor = hub_extractor

    @property
    def preset(self) -> str:
        return "struct" if self.hub_extractor is None else "standard"

    @property
    def cell_dim(self) -> int:
        pooled = 2 * self.hub_extractor.feature_dim if self.hub_extractor else 0
        return len(_CELL_SCALARS) + pooled

    @property
    def pair_dim(self) -> int:
        return 3 * self.cell_dim + 1 + len(_KIND_ORDER) + 1 + 2 + 1

    def cell_vector(self, cg: CellDependencyGraph, cell_id: str, dfg_graph=None) -> np.ndarray:
        data = cg.graph.nodes[cell_id]
        scalars = np.array(
            [
                len(data.get("defined_names", ())),
                len(data.get("free_uses", ())),
                len(data.get("undefined_uses", ())),
                float(data.get("has_opaque_effects", False)),
                float(data.get("parse_error", False)),
                np.log1p(data.get("n_lines", 0)),
                float(cg.graph.in_degree(cell_id)),
                float(cg.graph.out_degree(cell_id)),
            ],
            dtype=np.float32,
        )
        if self.hub_extractor is None:
            return scalars
        return np.concatenate([scalars, self._pooled_hub_vector(cell_id, dfg_graph)])

    def _pooled_hub_vector(self, cell_id: str, dfg_graph) -> np.ndarray:
        dim = self.hub_extractor.feature_dim
        vectors = []
        if dfg_graph is not None:
            for node_id, attrs in dfg_graph.nodes(data=True):
                if attrs.get("cell_id") == cell_id and attrs.get("node_type") == NODE_TYPES["CALL"]:
                    vec = self.hub_extractor.embed_node(attrs, 0.0, dfg_graph, node_id)
                    vectors.append(vec.detach().cpu().numpy().astype(np.float32))
        if not vectors:
            return np.zeros(2 * dim, dtype=np.float32)
        stack = np.stack(vectors)
        return np.concatenate([stack.mean(axis=0), stack.max(axis=0)])

    def pair_vector(self, cg: CellDependencyGraph, src: str, dst: str, dfg_graph=None, cell_cache: dict | None = None) -> np.ndarray:
        cache = cell_cache if cell_cache is not None else {}
        for cell in (src, dst):
            if cell not in cache:
                cache[cell] = self.cell_vector(cg, cell, dfg_graph)
        f_i, f_j = cache[src], cache[dst]

        nodes = cg.graph.nodes
        n = max(len(nodes), 2)
        doc_delta = (nodes[dst]["document_index"] - nodes[src]["document_index"]) / (n - 1)

        kind_onehot = np.zeros(len(_KIND_ORDER), dtype=np.float32)
        ambiguous = 0.0
        edge_data = cg.graph.get_edge_data(src, dst) or {}
        for data in edge_data.values():
            edge = data["edge"]
            kind_onehot[_KIND_ORDER.index(edge.kind.name)] = 1.0
            ambiguous = max(ambiguous, float(edge.ambiguous))
        reverse_edge = float(cg.graph.has_edge(dst, src))

        defs_i = nodes[src].get("defined_names", set())
        defs_j = nodes[dst].get("defined_names", set())
        uses_i = nodes[src].get("free_uses", set())
        uses_j = nodes[dst].get("free_uses", set())
        provides = _jaccard(defs_i, uses_j)
        consumes = _jaccard(defs_j, uses_i)

        return np.concatenate(
            [
                f_i,
                f_j,
                f_i - f_j,
                np.array([doc_delta], dtype=np.float32),
                kind_onehot,
                np.array([reverse_edge, provides, consumes, ambiguous], dtype=np.float32),
            ]
        )

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def generate_candidate_pairs(cg: CellDependencyGraph, window: int = 5, unrelated_cap: int = 30, seed: int = 42) -> list[tuple[str, str]]:
    """Ordered candidate pairs: every dependency edge, every ambiguous candidate x reader, plus a capped sample of unrelated near-document pairs."""
    pairs: set[tuple[str, str]] = set()
    for edge in cg.edges():
        pairs.add((edge.src_cell, edge.dst_cell))
        for candidate in edge.candidate_def_cells:
            if candidate != edge.dst_cell:
                pairs.add((candidate, edge.dst_cell))

    cells = cg.cells
    unrelated = []
    for i, a in enumerate(cells):
        for b in cells[i + 1 : i + 1 + window]:
            if (a, b) not in pairs and (b, a) not in pairs:
                unrelated.append((a, b))
    rng = random.Random(seed)
    rng.shuffle(unrelated)
    return sorted(pairs) + unrelated[:unrelated_cap]

def build_notebook_pairs(cells: list[dict], featurizer: CellPairFeaturizer, parser: GraphParser | None = None) -> tuple[np.ndarray, np.ndarray, bool] | None:
    """(features, labels, is_nonlinear) for one valid notebook, else None."""
    if not is_valid_supervision(cells):
        return None
    try:
        dfg = (parser or GraphParser()).parse(cells)
    except Exception as exc:
        logger.debug("Skipping notebook (parser failed: %s)", exc)
        return None

    cg = CellGraphProjector(dfg, cells).project()
    counts = {c["cell_id"]: c["execution_count"] for c in cells}
    dfg_graph = dfg.get_graph() if featurizer.hub_extractor else None

    xs, ys = [], []
    cell_cache: dict = {}
    for src, dst in generate_candidate_pairs(cg):
        if src not in counts or dst not in counts:
            continue
        label = 1.0 if counts[src] < counts[dst] else 0.0
        xs.append(featurizer.pair_vector(cg, src, dst, dfg_graph, cell_cache))
        ys.append(label)
        xs.append(featurizer.pair_vector(cg, dst, src, dfg_graph, cell_cache))
        ys.append(1.0 - label)

    if not xs:
        return None
    return np.stack(xs), np.array(ys, dtype=np.float32), not is_linear(cells)

def sample_linear_extension(cg: CellDependencyGraph, rng: random.Random) -> list[str]:
    """One uniform-ish random linear extension of the cell dependency graph (randomized Kahn)."""
    import networkx as nx

    simple = nx.DiGraph(cg.graph)
    indegree = dict(simple.in_degree())
    doc_index = {cell: data["document_index"] for cell, data in cg.graph.nodes(data=True)}
    ready = [c for c, d in indegree.items() if d == 0]
    order: list[str] = []
    emitted: set[str] = set()

    while len(order) < len(indegree):
        if not ready:
            ready.append(min((c for c in indegree if c not in emitted), key=doc_index.get))
        cell = ready.pop(rng.randrange(len(ready)))
        if cell in emitted:
            continue
        emitted.add(cell)
        order.append(cell)
        for succ in simple.successors(cell):
            if succ in emitted:
                continue
            indegree[succ] -= 1
            if indegree[succ] <= 0:
                ready.append(succ)
    return order

def is_valid_restoration_target(cells: list[dict], cg: CellDependencyGraph) -> bool:
    """Document order must be a plausible ground-truth execution order."""
    if len(cg.graph) < 4:
        return False
    if any(edge.out_of_order for edge in cg.edges()):
        return False
    if is_valid_supervision(cells) and not is_linear(cells):
        return False
    return True

def build_permuted_notebook_pairs(
    cells: list[dict],
    featurizer: CellPairFeaturizer,
    parser: GraphParser | None = None,
    n_permutations: int = 3,
    random_fraction: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Self-supervised pairs: features from a permuted projection of the notebook, labels from the original document order."""
    try:
        dfg = (parser or GraphParser()).parse(cells)
    except Exception as exc:
        logger.debug("Skipping notebook for permutation augmentation (parser failed: %s)", exc)
        return None

    projector_cache: dict = {}
    original_cg = CellGraphProjector(dfg, cells, cache=projector_cache).project()
    if not is_valid_restoration_target(cells, original_cg):
        return None

    original_pos = {cell: i for i, cell in enumerate(original_cg.cells)}
    cells_by_id = {cell.get("cell_id") or cell.get("id"): cell for cell in cells}
    dfg_graph = dfg.get_graph() if featurizer.hub_extractor else None
    rng = random.Random(seed)

    xs, ys = [], []
    for k in range(n_permutations):
        if rng.random() < random_fraction:
            permutation = list(original_pos)
            rng.shuffle(permutation)
        else:
            permutation = sample_linear_extension(original_cg, rng)
        if [original_pos[c] for c in permutation] == sorted(original_pos.values()):
            continue

        permuted_cells = [cells_by_id[cell_id] for cell_id in permutation if cell_id in cells_by_id]
        permuted_cg = CellGraphProjector(dfg, permuted_cells, cache=projector_cache).project()

        cell_cache: dict = {}
        for src, dst in generate_candidate_pairs(permuted_cg, seed=seed + k):
            if src not in original_pos or dst not in original_pos:
                continue
            label = 1.0 if original_pos[src] < original_pos[dst] else 0.0
            xs.append(featurizer.pair_vector(permuted_cg, src, dst, dfg_graph, cell_cache))
            ys.append(label)
            xs.append(featurizer.pair_vector(permuted_cg, dst, src, dfg_graph, cell_cache))
            ys.append(1.0 - label)

    if not xs:
        return None
    return np.stack(xs), np.array(ys, dtype=np.float32)

def build_selfsupervised_dataset(
    notebooks_dir: Path,
    featurizer: CellPairFeaturizer,
    n_permutations: int = 3,
    random_fraction: float = 0.3,
    limit: int | None = None,
    notebook_names: list[str] | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Permutation-restoration pairs over a notebook corpus."""
    if notebook_names is not None:
        paths = [notebooks_dir / name for name in notebook_names]
    else:
        paths = sorted(notebooks_dir.glob("*.ipynb"))
    if limit:
        paths = paths[:limit]

    parser = GraphParser()
    all_x, all_y, all_nb = [], [], []
    names: list[str] = []
    stats = {"scanned": 0, "eligible": 0}

    for path in paths:
        stats["scanned"] += 1
        cells = _cells_with_counts(path)
        if len(cells) < 4:
            continue
        built = build_permuted_notebook_pairs(cells, featurizer, parser, n_permutations=n_permutations, random_fraction=random_fraction, seed=seed + stats["scanned"])
        if built is None:
            continue
        x, y = built
        stats["eligible"] += 1
        nb_index = len(names)
        names.append(path.name)
        all_x.append(x)
        all_y.append(y)
        all_nb.append(np.full(len(y), nb_index, dtype=np.int32))

    logger.info(
        "Permutation-restoration dataset: %d/%d notebooks eligible, %d pairs (%d permutations each).",
        stats["eligible"],
        stats["scanned"],
        sum(len(y) for y in all_y),
        n_permutations,
    )
    if not all_x:
        return {"x": np.zeros((0, featurizer.pair_dim), dtype=np.float32), "y": np.zeros(0), "w": np.zeros(0), "notebook_id": np.zeros(0, dtype=np.int32), "notebooks": np.array([])}

    y = np.concatenate(all_y)
    return {
        "x": np.concatenate(all_x).astype(np.float32),
        "y": y,
        "w": np.ones(len(y), dtype=np.float32),
        "notebook_id": np.concatenate(all_nb),
        "notebooks": np.array(names),
    }

def build_dataset(
    notebooks_dir: Path,
    featurizer: CellPairFeaturizer,
    nonlinear_weight: float = 2.0,
    limit: int | None = None,
    notebook_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Scan *.ipynb under *notebooks_dir* into one pair dataset."""
    if notebook_names is not None:
        paths = [notebooks_dir / name for name in notebook_names]
    else:
        paths = sorted(notebooks_dir.glob("*.ipynb"))
    if limit:
        paths = paths[:limit]

    parser = GraphParser()
    all_x, all_y, all_w, all_nb = [], [], [], []
    names: list[str] = []
    stats = {"scanned": 0, "valid": 0, "nonlinear": 0}

    for path in paths:
        stats["scanned"] += 1
        cells = _cells_with_counts(path)
        if not cells:
            continue
        built = build_notebook_pairs(cells, featurizer, parser)
        if built is None:
            continue
        x, y, nonlinear = built
        stats["valid"] += 1
        stats["nonlinear"] += int(nonlinear)
        nb_index = len(names)
        names.append(path.name)
        all_x.append(x)
        all_y.append(y)
        all_w.append(np.full(len(y), nonlinear_weight if nonlinear else 1.0, dtype=np.float32))
        all_nb.append(np.full(len(y), nb_index, dtype=np.int32))

    logger.info(
        "Execution-order dataset: %d/%d notebooks valid (%d nonlinear), %d pairs.",
        stats["valid"],
        stats["scanned"],
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

def _cells_with_counts(path: Path) -> list[dict]:
    """Like NotebookExtractor.to_structured_cells, but keeps execution_count (the extractor drops it from its output)."""
    import nbformat

    try:
        with open(path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
    except Exception:
        return []
    cells = []
    for i, cell in enumerate(nb.cells):
        source = cell.get("source") or ""
        if cell.get("cell_type") != "code" or not source.strip():
            continue
        cells.append(
            {
                "cell_id": str(getattr(cell, "id", f"fallback-{i}")),
                "source": source,
                "execution_count": cell.get("execution_count"),
            }
        )
    return cells

def save_dataset(dataset: dict[str, np.ndarray], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset)
    meta = {
        "pairs": int(len(dataset["y"])),
        "notebooks": int(len(dataset["notebooks"])),
        "positive_rate": float(dataset["y"].mean()) if len(dataset["y"]) else 0.0,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
