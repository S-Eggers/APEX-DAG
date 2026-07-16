from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from enum import Flag, auto

import networkx as nx
import numpy as np
import torch

from SystemX.nn.data.v2.fasttext_embedding import FastTextEmbeddingV2
from SystemX.sca.constants import COMPUTE_HUBS, NODE_TYPES

logger = logging.getLogger(__name__)

class FeatureGroup(Flag):
    """Bit-flag enum - combine with | to select feature subsets."""

    CODE_EMB = auto()
    API_EMB = auto()
    LIB_EMB = auto()
    DEGREE = auto()
    CELL_POS = auto()
    NEIGHBOR = auto()
    LIB_CATEGORY = auto()
    LINEAGE_STRUCT = auto()
    SEMANTIC_ANCHOR = auto()
    CENTRALITY = auto()

    STANDARD = CODE_EMB | DEGREE | CELL_POS | NEIGHBOR | LINEAGE_STRUCT
    """Default: code embedding + pure graph structure (315-d with FastText-300).
    The OOD-honest set - no API/lib embedding, no library category, no API anchors."""

    EMB_RICH = CODE_EMB | API_EMB | LIB_EMB | DEGREE | CELL_POS | NEIGHBOR | LIB_CATEGORY | LINEAGE_STRUCT | SEMANTIC_ANCHOR
    """Former default (pre-this-change): adds API-name + library embeddings, library
    category and API-name anchors on top of STANDARD (924-d). Higher in-distribution
    CV, but the extra signals are the shortcut that fails OOD - kept for comparison."""

    EMBEDDING_ONLY = CODE_EMB | API_EMB | LIB_EMB
    """Semantic embeddings only, no structural features. 900-d."""

    STRUCTURAL_ONLY = DEGREE | CELL_POS | NEIGHBOR | LINEAGE_STRUCT
    """Pure graph topology, no text and no API-name heuristics."""

    ALL = EMB_RICH | CENTRALITY
    """All feature groups (EMB_RICH + centrality)."""

    API29 = API_EMB | DEGREE | CELL_POS | NEIGHBOR | LIB_CATEGORY | LINEAGE_STRUCT | SEMANTIC_ANCHOR | CENTRALITY
    """API-name embedding + all 29 non-embedding features (degree, cell position,
    neighbor, library category, lineage topology, semantic anchors, centrality) -
    329-d. This is ``ALL`` minus the two 300-d code and library *embeddings*: it
    keeps only the cheap API-name embedding, testing whether the full 929-d
    accuracy survives without the expensive code+lib FastText vectors. NB: it
    still contains the library-category shortcut, so it is not the OOD-honest set
    (that is ``STANDARD``)."""

    API24 = API_EMB | DEGREE | CELL_POS | NEIGHBOR | LIB_CATEGORY | LINEAGE_STRUCT | SEMANTIC_ANCHOR
    """``API29`` minus ``CENTRALITY`` - API-name embedding + the 24 non-embedding
    features except the (expensive) graph-centrality group - 324-d. Tests whether
    api29's out-of-distribution win survives without the centrality features that make
    it ~3x slower than the 315-d standard set."""

FEATURE_PRESETS: dict[str, FeatureGroup] = {
    "standard": FeatureGroup.STANDARD,
    "emb_rich": FeatureGroup.EMB_RICH,
    "all": FeatureGroup.ALL,
    "emb_only": FeatureGroup.CODE_EMB,
    "api_lib": FeatureGroup.API_EMB | FeatureGroup.LIB_EMB,
    "api29": FeatureGroup.API29,
    "api24": FeatureGroup.API24,
    "struct_only": FeatureGroup.STRUCTURAL_ONLY,
}

_VARIABLE_TYPE = NODE_TYPES.get("VARIABLE", 0)
_LITERAL_TYPE = NODE_TYPES.get("LITERAL", 8)
_INTERMEDIATE_TYPE = NODE_TYPES.get("INTERMEDIATE", 4)

_EMB_DIM = 300

_LIB_FAMILIES: list[tuple[str, frozenset[str]]] = [
    ("data_io", frozenset({"pandas", "pd", "numpy", "np", "csv", "json", "sql", "sqlite", "sqlalchemy", "requests",
                           "urllib", "h5py", "scipy", "openpyxl", "pyarrow", "arrow", "mdanalysis", "nglview", "datasets", "boto3", "glob",
                           "polars", "pl", "pyspark", "spark", "dask", "dd", "modin", "cudf", "vaex",
                           "koalas", "ks", "beam", "apache_beam", "duckdb", "ibis", "datatable", "fireducks"})),
    ("ml_classic", frozenset({"sklearn", "xgboost", "xgb", "lightgbm", "lgb", "catboost", "statsmodels", "mltools", "linear", "svm", "ensemble"})),
    ("deep_learning", frozenset({"torch", "tensorflow", "tf", "keras", "transformers", "jax", "fastai", "lightning"})),
    ("viz", frozenset({"matplotlib", "pyplot", "plt", "seaborn", "sns", "plotly", "bokeh", "altair"})),
    ("persistence", frozenset({"pickle", "joblib", "dill"})),
    ("sys_env", frozenset({"os", "sys", "logging", "warnings", "gc", "argparse", "subprocess", "shutil"})),
]
_N_LIB_FAMILIES = len(_LIB_FAMILIES)

def _lib_category_vector(base_inputs: str, label: str, code: str = "") -> np.ndarray:
    """Binary indicator over _LIB_FAMILIES from a node's library provenance."""
    tokens = set(re.split(r"[^a-z0-9_]+", f"{base_inputs} {label} {code}".lower()))
    return np.array([1.0 if (kws & tokens) else 0.0 for _, kws in _LIB_FAMILIES], dtype=np.float32)

_MODEL_APIS = frozenset({
    "fit", "fit_predict", "predict", "predict_proba", "predict_log_proba", "score",
    "cross_val_score", "cross_validate", "evaluate", "partial_fit",
})
_N_LINEAGE = 10
_N_SEMANTIC = 3
_N_CENTRALITY = 5

_GROUP_NAMES: dict[str, str] = {
    "code_emb": "Code embedding",
    "api_emb": "API-name embedding",
    "lib_emb": "Library-context embedding",
    "degree": "Node degree",
    "cell_pos": "Cell position",
    "neighbor": "Neighbor counts",
    "lib_category": "Library category",
    "lineage_struct": "Graph structure",
    "semantic_anchor": "Semantic anchors (API heuristics)",
    "centrality": "Graph centrality",
}
_GROUP_DESCRIPTIONS: dict[str, str] = {
    "code_emb": "Text embedding of the call's full source snippet.",
    "api_emb": "Text embedding of the extracted API/method name.",
    "lib_emb": "Text embedding of the library provenance (base_inputs).",
    "degree": "In/out degree of the node in the dataflow graph.",
    "cell_pos": "Normalized position of the node's cell within the notebook.",
    "neighbor": "Counts of variable and literal predecessor nodes.",
    "lib_category": "Which library family the call draws from.",
    "lineage_struct": "Name-independent graph structure: source/sink role, loader fingerprint, distance from a data source, and output reuse.",
    "semantic_anchor": "API-name heuristics: reachability from a load call (read_/scan_/load_) or to a model call (fit/predict).",
    "centrality": "Graph centrality measures (betweenness, pagerank, ...).",
}
_SCALAR_NAMES: dict[str, list[str]] = {
    "degree": ["in_degree", "out_degree"],
    "cell_pos": ["cell_position"],
    "neighbor": ["n_variable_predecessors", "n_literal_predecessors"],
    "lib_category": [f"lib_{name}" for name, _ in _LIB_FAMILIES],
    "lineage_struct": ["is_sink", "n_successor_ops_norm", "is_source", "path_literal_input", "struct_loader",
                       "from_data_source", "n_source_ancestors_norm", "depth_from_source_norm", "output_reuse_norm",
                       "in_assign"],
    "semantic_anchor": ["from_load", "to_model", "supervised_fit"],
    "centrality": ["in_degree_centrality", "out_degree_centrality", "betweenness", "pagerank", "clustering"],
}
_SCALAR_DESCRIPTIONS: dict[str, str] = {
    "in_degree": "number of incoming dataflow edges",
    "out_degree": "number of outgoing dataflow edges",
    "cell_position": "normalized notebook cell ordinal (0=first, 1=last)",
    "n_variable_predecessors": "number of variable inputs feeding this call",
    "n_literal_predecessors": "number of literal (constant) inputs feeding this call",
    "lib_data_io": "uses a data-I/O library (pandas, numpy, csv, ...)",
    "lib_ml_classic": "uses a classic-ML library (sklearn, xgboost, ...)",
    "lib_deep_learning": "uses a deep-learning library (torch, tensorflow, ...)",
    "lib_viz": "uses a visualization library (matplotlib, seaborn, ...)",
    "lib_persistence": "uses a serialization library (pickle, joblib, ...)",
    "lib_sys_env": "uses a system/env library (os, sys, logging, ...)",
    "is_sink": "terminal node (no outgoing edges)",
    "n_successor_ops_norm": "normalized count of downstream operations",
    "is_source": "source call - no variable/dataframe inputs feed it",
    "path_literal_input": "fed by a data-file path or URI literal (e.g. 'data.csv', 's3://...')",
    "from_data_source": "reachable downstream from a structural loader (name-independent)",
    "n_source_ancestors_norm": "normalized count of distinct data-source ancestors (join/merge >= 2)",
    "depth_from_source_norm": "normalized hop distance from the nearest data source",
    "output_reuse_norm": "normalized number of downstream consumers of this call's output",
    "in_assign": "statement context: 1.0 assigned/mutated (writes state), 0.0 bare expression (displayed), 0.5 undetermined",
    "from_load": "reachable downstream from an API-named load call (read_/scan_/load_)",
    "to_model": "can reach a model-fitting call (fit/predict/score/...)",
    "supervised_fit": "supervised fit(X, y) signature (has a target arg)",
    "struct_loader": "loader fingerprint: source call fed only by a data-path literal (name-independent)",
    "in_degree_centrality": "normalized in-degree centrality",
    "out_degree_centrality": "normalized out-degree centrality",
    "betweenness": "betweenness centrality",
    "pagerank": "PageRank score",
    "clustering": "clustering coefficient",
}

@dataclass(frozen=True)
class FeatureSpan:
    """One contiguous slice of a feature vector belonging to a single group."""

    key: str
    name: str
    description: str
    start: int
    end: int
    scalar_names: list[str] | None

def _node_api(attrs: dict) -> str:
    parts = str(attrs.get("label", "")).strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() in ("call", "method"):
        return parts[1].strip().lower()
    return _extract_api_name(str(attrs.get("label", "")), str(attrs.get("code", "")))

def _is_load_api(api: str) -> bool:
    return api.startswith(("read_", "scan_", "fetch_", "load_")) or api in {"loadtxt", "genfromtxt", "imread", "load"}

_DATA_PATH_RE = re.compile(
    r"\.(?:csv|tsv|parquet|parq|json|jsonl|ndjson|feather|orc|xls[xmb]?|"
    r"h5|hdf5|pkl|pickle|npy|npz|txt|dat|avro|arrow|db|sqlite3?|gz|zip)\b"
    r"|(?:https?|s3|gs|gcs|hdfs|file|abfss?|wasbs?|dbfs)://"
    r"|/[\w.\-]+\.\w+",
    re.IGNORECASE,
)

def _looks_like_data_path(text: str) -> bool:
    return bool(text) and _DATA_PATH_RE.search(text) is not None

def _predecessor_shape(graph: nx.MultiDiGraph, node_id: object) -> tuple[int, bool]:
    """(#variable predecessors, has-a-data-path-literal predecessor) for a CALL node."""
    n_var = 0
    has_path_lit = False
    for p in graph.predecessors(node_id):
        pa = graph.nodes[p]
        ntype = int(pa.get("node_type", -1))
        if ntype == _VARIABLE_TYPE:
            n_var += 1
        elif ntype == _LITERAL_TYPE and not has_path_lit:
            txt = str(pa.get("code", "") or pa.get("label", "") or pa.get("value", ""))
            if _looks_like_data_path(txt):
                has_path_lit = True
    return n_var, has_path_lit

def _is_structural_loader(n_var_preds: int, has_path_literal: bool) -> bool:
    """Loader *fingerprint*: a source call (no variable/dataframe inputs) fed by a data-path literal."""
    return n_var_preds == 0 and has_path_literal

def _structural_source_maps(graph: nx.MultiDiGraph) -> tuple[set, dict, dict]:
    """Name-independent data-source reachability."""
    sources: set = set()
    for n, a in graph.nodes(data=True):
        if int(a.get("node_type", -1)) not in COMPUTE_HUBS:
            continue
        n_var, has_path = _predecessor_shape(graph, n)
        if _is_structural_loader(n_var, has_path):
            sources.add(n)

    from collections import defaultdict, deque

    from_source: set = set()
    n_source_ancestors: dict = defaultdict(int)
    for s in sources:
        desc = nx.descendants(graph, s)
        from_source |= desc | {s}
        for d in desc:
            n_source_ancestors[d] += 1

    depth_from_source: dict = {s: 0 for s in sources}
    frontier: deque = deque(sources)
    while frontier:
        u = frontier.popleft()
        for _, v in graph.out_edges(u):
            if v not in depth_from_source:
                depth_from_source[v] = depth_from_source[u] + 1
                frontier.append(v)
    return from_source, dict(n_source_ancestors), depth_from_source

def _semantic_anchor_sets(graph: nx.MultiDiGraph) -> tuple[set, set]:
    """API-NAME-seeded reachability: (reachable from a load call, can reach a model call)."""
    loads, models = set(), set()
    for n, a in graph.nodes(data=True):
        if int(a.get("node_type", -1)) not in COMPUTE_HUBS:
            continue
        api = _node_api(a)
        if _is_load_api(api):
            loads.add(n)
        if api in _MODEL_APIS:
            models.add(n)
    from_load: set = set()
    for s in loads:
        from_load |= nx.descendants(graph, s) | {s}
    to_model: set = set()
    for m in models:
        to_model |= nx.ancestors(graph, m) | {m}
    return from_load, to_model

def _assign_context_map(graph: nx.MultiDiGraph) -> dict:
    """Per-CALL statement context: is this call part of an ASSIGN/mutation or a bare EXPR?"""
    from collections import deque

    ctx: dict = {}
    for n, a in graph.nodes(data=True):
        if int(a.get("node_type", -1)) not in COMPUTE_HUBS:
            continue
        seen: set = set()
        dq: deque = deque(v for _, v in graph.out_edges(n))
        verdict = 0.5
        while dq:
            s = dq.popleft()
            if s in seen:
                continue
            seen.add(s)
            nt = int(graph.nodes[s].get("node_type", -1))
            if nt == _VARIABLE_TYPE:
                verdict = 1.0
                break
            if nt == _INTERMEDIATE_TYPE and str(s).startswith("sink_"):
                verdict = 0.0
                break
            if nt in COMPUTE_HUBS:
                dq.extend(v for _, v in graph.out_edges(s))
        ctx[n] = verdict
    return ctx

def _lineage_struct_vector(graph: nx.MultiDiGraph, node_id: object, out_deg: dict,
                           from_source: set, n_source_ancestors: dict, depth_from_source: dict,
                           assign_ctx: dict) -> np.ndarray:
    """Name-independent graph-structure features for one CALL node."""
    n_succ_ops = sum(
        1
        for _, v in graph.out_edges(node_id)
        for _, w in graph.out_edges(v)
        if int(graph.nodes[w].get("node_type", -1)) in COMPUTE_HUBS
    )
    output_reuse = sum(graph.out_degree(v) for _, v in graph.out_edges(node_id))
    n_var_preds, has_path_lit = _predecessor_shape(graph, node_id)
    return np.array([
        1.0 if out_deg.get(node_id, 0) == 0 else 0.0,
        min(n_succ_ops / 5.0, 1.0),
        1.0 if n_var_preds == 0 else 0.0,
        1.0 if has_path_lit else 0.0,
        1.0 if _is_structural_loader(n_var_preds, has_path_lit) else 0.0,
        1.0 if node_id in from_source else 0.0,
        min(n_source_ancestors.get(node_id, 0) / 3.0, 1.0),
        min(depth_from_source.get(node_id, 0) / 8.0, 1.0),
        min(output_reuse / 5.0, 1.0),
        assign_ctx.get(node_id, 0.5),
    ], dtype=np.float32)

def _semantic_anchor_vector(attrs: dict, node_id: object, from_load: set, to_model: set) -> np.ndarray:
    """API-name-derived anchor features for one CALL node (read_/scan_/fit/predict)."""
    api = _node_api(attrs)
    code = str(attrs.get("code", ""))
    supervised_fit = 1.0 if api in {"fit", "fit_predict"} and re.search(r"\.fit\w*\s*\([^)]*,", code) else 0.0
    return np.array([
        1.0 if node_id in from_load else 0.0,
        1.0 if node_id in to_model else 0.0,
        supervised_fit,
    ], dtype=np.float32)

def _centrality_maps(graph: nx.MultiDiGraph) -> dict[str, dict]:
    """Per-node centrality measures (computed once per graph)."""
    n = graph.number_of_nodes()
    try:
        simple = nx.DiGraph(graph)
        betw = nx.betweenness_centrality(simple) if n > 2 else {}
        pr = nx.pagerank(simple, max_iter=100) if n > 0 else {}
        clus = nx.clustering(simple)
    except Exception:
        betw, pr, clus = {}, {}, {}
    norm = max(n - 1, 1)
    return {
        "in_dc": {x: graph.in_degree(x) / norm for x in graph.nodes()},
        "out_dc": {x: graph.out_degree(x) / norm for x in graph.nodes()},
        "betw": betw,
        "pr": pr,
        "clus": clus,
    }

def infer_feature_groups(in_features: int) -> FeatureGroup:
    """Map a feature-vector dimension back to the FeatureGroup that produced it."""
    d = _EMB_DIM
    lib = _N_LIB_FAMILIES
    lin = _N_LINEAGE
    sem = _N_SEMANTIC
    cen = _N_CENTRALITY
    candidates: list[tuple[int, FeatureGroup]] = [
        (d + 2 + 1 + 2 + lin, FeatureGroup.STANDARD),
        (3 * d + 2 + 1 + 2 + lib + lin + sem, FeatureGroup.EMB_RICH),
        (3 * d + 2 + 1 + 2 + lib + lin + sem + cen, FeatureGroup.ALL),
        (3 * d, FeatureGroup.EMBEDDING_ONLY),
        (2 + 1 + 2 + lin, FeatureGroup.STRUCTURAL_ONLY),
        (2 * d, FeatureGroup.API_EMB | FeatureGroup.LIB_EMB),
        (d + 2 + 1 + 2 + lib + lin + sem + cen, FeatureGroup.API29),
        (d + 2 + 1 + 2 + lib + lin + sem, FeatureGroup.API24),
        (d, FeatureGroup.CODE_EMB),
        (d + 2, FeatureGroup.CODE_EMB | FeatureGroup.DEGREE),
        (d + 2 + 1, FeatureGroup.CODE_EMB | FeatureGroup.DEGREE | FeatureGroup.CELL_POS),
        (d + 2 + 1 + 2, FeatureGroup.CODE_EMB | FeatureGroup.DEGREE | FeatureGroup.CELL_POS | FeatureGroup.NEIGHBOR),
    ]
    for dim, fg in candidates:
        if dim == in_features:
            return fg
    return FeatureGroup.STANDARD

def _extract_api_name(label: str, code: str) -> str:
    """Best-effort extraction of the bare API/method name."""
    parts = label.strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() in ("call", "method"):
        return parts[1]

    try:
        tree = ast.parse(code.strip(), mode="eval")
        if isinstance(tree.body, ast.Call):
            func = tree.body.func
            if isinstance(func, ast.Name):
                return func.id
            if isinstance(func, ast.Attribute):
                return func.attr
    except (SyntaxError, ValueError):
        pass

    m = re.search(r"\b([A-Za-z_]\w*)\s*\(", code)
    return m.group(1) if m else code

def _cell_ordinal(node_cell_id: str, ordered_cells: list[str]) -> float:
    """Normalised 0.0...1.0 position of this node's cell in the notebook."""
    if not ordered_cells:
        return 0.0
    try:
        idx = ordered_cells.index(node_cell_id)
    except ValueError:
        return 0.5
    return idx / max(len(ordered_cells) - 1, 1)

class ComputeHubFeatureExtractor:
    """Extracts a fixed-length feature vector for every COMPUTE_HUB (CALL) node."""

    def __init__(
        self,
        embedding: FastTextEmbeddingV2 | None = None,
        groups: FeatureGroup = FeatureGroup.STANDARD,
    ) -> None:
        self._embedding = embedding or FastTextEmbeddingV2()
        self.groups = groups

    @property
    def feature_dim(self) -> int:
        dim = 0
        emb_d = self._embedding.dimension
        if FeatureGroup.CODE_EMB in self.groups:
            dim += emb_d
        if FeatureGroup.API_EMB in self.groups:
            dim += emb_d
        if FeatureGroup.LIB_EMB in self.groups:
            dim += emb_d
        if FeatureGroup.DEGREE in self.groups:
            dim += 2
        if FeatureGroup.CELL_POS in self.groups:
            dim += 1
        if FeatureGroup.NEIGHBOR in self.groups:
            dim += 2
        if FeatureGroup.LIB_CATEGORY in self.groups:
            dim += _N_LIB_FAMILIES
        if FeatureGroup.LINEAGE_STRUCT in self.groups:
            dim += _N_LINEAGE
        if FeatureGroup.SEMANTIC_ANCHOR in self.groups:
            dim += _N_SEMANTIC
        if FeatureGroup.CENTRALITY in self.groups:
            dim += _N_CENTRALITY
        return dim

    @property
    def description(self) -> str:
        parts = []
        if FeatureGroup.CODE_EMB in self.groups:
            parts.append(f"code_emb({self._embedding.dimension}d)")
        if FeatureGroup.API_EMB in self.groups:
            parts.append(f"api_emb({self._embedding.dimension}d)")
        if FeatureGroup.LIB_EMB in self.groups:
            parts.append(f"lib_emb({self._embedding.dimension}d)")
        if FeatureGroup.DEGREE in self.groups:
            parts.append("degree(2)")
        if FeatureGroup.CELL_POS in self.groups:
            parts.append("cell_pos(1)")
        if FeatureGroup.NEIGHBOR in self.groups:
            parts.append("neighbor(2)")
        if FeatureGroup.LIB_CATEGORY in self.groups:
            parts.append(f"lib_cat({_N_LIB_FAMILIES})")
        if FeatureGroup.LINEAGE_STRUCT in self.groups:
            parts.append(f"lineage({_N_LINEAGE})")
        if FeatureGroup.SEMANTIC_ANCHOR in self.groups:
            parts.append(f"semantic({_N_SEMANTIC})")
        if FeatureGroup.CENTRALITY in self.groups:
            parts.append(f"centrality({_N_CENTRALITY})")
        return "+".join(parts) + f" = {self.feature_dim}d"

    def feature_layout(self) -> list[FeatureSpan]:
        """Ordered spans mapping every feature index to a named group."""
        emb_d = self._embedding.dimension
        spans: list[FeatureSpan] = []
        cursor = 0

        def _add(key: str, size: int, scalar_names: list[str] | None) -> None:
            nonlocal cursor
            spans.append(
                FeatureSpan(
                    key=key,
                    name=_GROUP_NAMES[key],
                    description=_GROUP_DESCRIPTIONS[key],
                    start=cursor,
                    end=cursor + size,
                    scalar_names=scalar_names,
                )
            )
            cursor += size

        if FeatureGroup.CODE_EMB in self.groups:
            _add("code_emb", emb_d, None)
        if FeatureGroup.API_EMB in self.groups:
            _add("api_emb", emb_d, None)
        if FeatureGroup.LIB_EMB in self.groups:
            _add("lib_emb", emb_d, None)
        if FeatureGroup.DEGREE in self.groups:
            _add("degree", 2, _SCALAR_NAMES["degree"])
        if FeatureGroup.CELL_POS in self.groups:
            _add("cell_pos", 1, _SCALAR_NAMES["cell_pos"])
        if FeatureGroup.NEIGHBOR in self.groups:
            _add("neighbor", 2, _SCALAR_NAMES["neighbor"])
        if FeatureGroup.LIB_CATEGORY in self.groups:
            _add("lib_category", _N_LIB_FAMILIES, _SCALAR_NAMES["lib_category"])
        if FeatureGroup.LINEAGE_STRUCT in self.groups:
            _add("lineage_struct", _N_LINEAGE, _SCALAR_NAMES["lineage_struct"])
        if FeatureGroup.SEMANTIC_ANCHOR in self.groups:
            _add("semantic_anchor", _N_SEMANTIC, _SCALAR_NAMES["semantic_anchor"])
        if FeatureGroup.CENTRALITY in self.groups:
            _add("centrality", _N_CENTRALITY, _SCALAR_NAMES["centrality"])
        return spans

    def extract(
        self,
        graph: nx.MultiDiGraph,
    ) -> tuple[np.ndarray, list[object], list[int]]:
        """Parameters ---------- graph : raw DFG or annotation-derived MultiDiGraph."""
        in_deg = dict(graph.in_degree())
        out_deg = dict(graph.out_degree())

        from_source: set = set()
        n_source_ancestors: dict = {}
        depth_from_source: dict = {}
        assign_ctx: dict = {}
        if FeatureGroup.LINEAGE_STRUCT in self.groups:
            from_source, n_source_ancestors, depth_from_source = _structural_source_maps(graph)
            assign_ctx = _assign_context_map(graph)

        from_load: set = set()
        to_model: set = set()
        if FeatureGroup.SEMANTIC_ANCHOR in self.groups:
            from_load, to_model = _semantic_anchor_sets(graph)

        centrality = _centrality_maps(graph) if FeatureGroup.CENTRALITY in self.groups else {}

        seen_cells: list[str] = []
        for _, attrs in graph.nodes(data=True):
            cid = attrs.get("cell_id", "")
            if cid and cid not in seen_cells:
                seen_cells.append(cid)

        X_rows: list[np.ndarray] = []
        node_ids: list[object] = []
        labels: list[int] = []

        for node_id, attrs in graph.nodes(data=True):
            if int(attrs.get("node_type", -1)) not in COMPUTE_HUBS:
                continue

            code = str(attrs.get("code", "") or attrs.get("label", ""))
            label_str = str(attrs.get("label", ""))
            base_inputs = str(attrs.get("base_inputs", "") or "")
            cell_id = str(attrs.get("cell_id", ""))

            parts: list[np.ndarray] = []

            if FeatureGroup.CODE_EMB in self.groups:
                parts.append(self._embedding.embed(code).numpy())

            if FeatureGroup.API_EMB in self.groups:
                api_name = _extract_api_name(label_str, code)
                parts.append(self._embedding.embed(api_name).numpy())

            if FeatureGroup.LIB_EMB in self.groups:
                parts.append(self._embedding.embed(base_inputs).numpy())

            if FeatureGroup.DEGREE in self.groups:
                parts.append(np.array([in_deg.get(node_id, 0), out_deg.get(node_id, 0)], dtype=np.float32))

            if FeatureGroup.CELL_POS in self.groups:
                parts.append(np.array([_cell_ordinal(cell_id, seen_cells)], dtype=np.float32))

            if FeatureGroup.NEIGHBOR in self.groups:
                preds = list(graph.predecessors(node_id))
                n_var = sum(1 for p in preds if graph.nodes[p].get("node_type") == _VARIABLE_TYPE)
                n_lit = sum(1 for p in preds if graph.nodes[p].get("node_type") == _LITERAL_TYPE)
                parts.append(np.array([n_var, n_lit], dtype=np.float32))

            if FeatureGroup.LIB_CATEGORY in self.groups:
                parts.append(_lib_category_vector(base_inputs, label_str, code))

            if FeatureGroup.LINEAGE_STRUCT in self.groups:
                parts.append(_lineage_struct_vector(graph, node_id, out_deg, from_source, n_source_ancestors, depth_from_source, assign_ctx))

            if FeatureGroup.SEMANTIC_ANCHOR in self.groups:
                parts.append(_semantic_anchor_vector(attrs, node_id, from_load, to_model))

            if FeatureGroup.CENTRALITY in self.groups:
                parts.append(np.array([centrality[k].get(node_id, 0.0) for k in ("in_dc", "out_dc", "betw", "pr", "clus")], dtype=np.float32))

            X_rows.append(np.concatenate(parts))
            node_ids.append(node_id)
            labels.append(int(attrs.get("domain_label", -1)))

        X = np.stack(X_rows) if X_rows else np.empty((0, self.feature_dim), dtype=np.float32)
        return X, node_ids, labels

    def embed_node(self, attrs: dict, cell_pos: float, graph: nx.MultiDiGraph, node_id: object) -> torch.Tensor:
        """Single-node feature vector as a tensor (used by MLPLabeler at inference time)."""
        x, _, _ = self.extract(graph)
        _, ids, _ = self.extract(graph)
        try:
            idx = ids.index(node_id)
            return torch.tensor(x[idx], dtype=torch.float32)
        except (ValueError, IndexError):
            return torch.zeros(self.feature_dim, dtype=torch.float32)
