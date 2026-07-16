from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum

import networkx as nx

from SystemX.sca.constants import COMPUTE_HUBS, EDGE_TYPES
from SystemX.sca.refinement.constants import ATTR_NODE_TYPE

_CALLER = EDGE_TYPES["CALLER"]
_INPUT = EDGE_TYPES["INPUT"]
_OUTPUT = EDGE_TYPES["OMITTED"]

def _leaf_callee(code: object) -> str:
    """The method/function of the *outermost* call in code."""
    s = str(code or "")
    depth = 0
    last = ""
    for i, ch in enumerate(s):
        if ch == "(":
            if depth == 0:
                j = i - 1
                while j >= 0 and s[j].isspace():
                    j -= 1
                k = j
                while k >= 0 and (s[k].isalnum() or s[k] == "_"):
                    k -= 1
                name = s[k + 1:j + 1]
                if name:
                    last = name
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
    return last

class LeakageClass(StrEnum):
    """The defect classes the static analyser can decide."""

    PREPROCESSING_BEFORE_SPLIT = "preprocessing_before_split"
    TARGET_LEAKAGE = "target_leakage"
    TEST_IN_TRAIN = "test_in_train"
    METRIC_ON_TRAIN = "metric_on_train"
    NO_HOLDOUT_EVALUATION = "no_holdout_evaluation"

_SPLIT_CALLEES = frozenset({"train_test_split"})
_SPLITTER_CTOR_SUFFIXES = ("KFold", "ShuffleSplit", "LeaveOneOut", "LeavePOut")
_SPLITTER_CTOR_NAMES = frozenset({"TimeSeriesSplit", "PredefinedSplit"})

_FIT_CALLEES = frozenset({"fit", "fit_transform", "fit_resample"})
_TRAIN_CALLEES = frozenset({"fit", "fit_predict", "partial_fit", "fit_resample"})
_INFERENCE_CALLEES = frozenset({"predict", "predict_proba", "predict_log_proba", "transform", "score", "decision_function"})

_TRANSFORMER_CTOR_SUFFIXES = (
    "Scaler", "Encoder", "Imputer", "Normalizer", "Binarizer", "Transformer",
    "Vectorizer", "Discretizer", "Selector",
)
_TRANSFORMER_CTOR_NAMES = frozenset({
    "PCA", "TruncatedSVD", "SelectKBest", "SelectPercentile", "PolynomialFeatures",
    "SMOTE", "ADASYN", "RandomOverSampler", "RandomUnderSampler", "TfidfVectorizer",
    "CountVectorizer", "LabelEncoder", "OneHotEncoder", "OrdinalEncoder",
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "QuantileTransformer", "PowerTransformer", "KBinsDiscretizer", "Normalizer",
    "SimpleImputer", "KNNImputer", "IterativeImputer", "VarianceThreshold",
})
_MODEL_CTOR_SUFFIXES = ("Classifier", "Regressor", "Regression", "SVC", "SVR")
_MODEL_CTOR_NAMES = frozenset({
    "LinearRegression", "LogisticRegression", "KMeans", "KNeighborsClassifier",
    "KNeighborsRegressor", "GaussianNB", "MultinomialNB", "XGBClassifier",
    "XGBRegressor", "LGBMClassifier", "LGBMRegressor", "CatBoostClassifier",
    "MLPClassifier", "MLPRegressor",
})

_EVAL_CALLEES = frozenset({
    "score", "accuracy_score", "f1_score", "precision_score", "recall_score",
    "roc_auc_score", "average_precision_score", "log_loss", "mean_squared_error",
    "mean_absolute_error", "r2_score", "confusion_matrix", "classification_report",
    "balanced_accuracy_score", "matthews_corrcoef", "cohen_kappa_score",
})

_TEST_TOKENS = frozenset({"test", "val", "valid", "validation", "holdout", "eval"})
_TRAIN_TOKENS = frozenset({"train", "training", "fit"})

def _tokens(name: object) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t}

@dataclass(frozen=True)
class LeakageFinding:
    """One decided defect, localised to the offending operation node."""

    leakage_class: str
    node: str
    code: str
    detail: str
    source_node: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["leakage_class"] = str(self.leakage_class)
        return d

@dataclass
class _Ctx:
    """Precomputed views over the graph shared by all detectors."""

    g: nx.MultiDiGraph
    order: dict[str, int] = field(default_factory=dict)
    hubs: list[str] = field(default_factory=list)

    def inputs(self, hub: str) -> list[str]:
        return [u for u, _, d in self.g.in_edges(hub, data=True) if d.get("edge_type") == _INPUT]

    def outputs(self, hub: str) -> list[str]:
        return [v for _, v, d in self.g.out_edges(hub, data=True) if d.get("edge_type") == _OUTPUT]

    def receiver(self, hub: str) -> str | None:
        for u, _, d in self.g.in_edges(hub, data=True):
            if d.get("edge_type") == _CALLER:
                return u
        return None

    def producer(self, var: str) -> str | None:
        """The hub that produced var (its OMITTED-typed predecessor)."""
        for u, _, d in self.g.in_edges(var, data=True):
            if d.get("edge_type") == _OUTPUT and self.g.nodes[u].get(ATTR_NODE_TYPE) in COMPUTE_HUBS:
                return u
        return None

    def receiver_ctor(self, hub: str) -> str:
        """Callee name of the constructor of hub's receiver object."""
        recv = self.receiver(hub)
        if recv is None:
            return ""
        producer = self.producer(recv)
        if producer is None:
            return _leaf_callee(self.g.nodes[recv].get("code", ""))
        return _leaf_callee(self.g.nodes[producer].get("code", ""))

def _is_transformer(ctor: str) -> bool:
    return ctor in _TRANSFORMER_CTOR_NAMES or ctor.endswith(_TRANSFORMER_CTOR_SUFFIXES)

def _is_model(ctor: str) -> bool:
    return ctor in _MODEL_CTOR_NAMES or ctor.endswith(_MODEL_CTOR_SUFFIXES)

def _build_ctx(g: nx.MultiDiGraph) -> _Ctx:
    order = {n: i for i, n in enumerate(g.nodes())}
    hubs = [n for n, d in g.nodes(data=True) if d.get(ATTR_NODE_TYPE) in COMPUTE_HUBS]
    return _Ctx(g=g, order=order, hubs=hubs)

def _callee(ctx: _Ctx, hub: str) -> str:
    return _leaf_callee(ctx.g.nodes[hub].get("code", ""))

def _split_hubs(ctx: _Ctx) -> list[str]:
    out = []
    for h in ctx.hubs:
        callee = _callee(ctx, h)
        if callee in _SPLIT_CALLEES:
            out.append(h)
        elif callee == "split":
            ctor = ctx.receiver_ctor(h)
            if ctor in _SPLITTER_CTOR_NAMES or ctor.endswith(_SPLITTER_CTOR_SUFFIXES):
                out.append(h)
    return out

def _transformer_fit_hubs(ctx: _Ctx) -> list[str]:
    out = []
    for h in ctx.hubs:
        callee = _callee(ctx, h)
        if callee not in _FIT_CALLEES:
            continue
        if callee in ("fit_transform", "fit_resample"):
            out.append(h)
        elif callee == "fit" and _is_transformer(ctx.receiver_ctor(h)):
            out.append(h)
    return out

def _model_fit_hubs(ctx: _Ctx) -> list[str]:
    out = []
    for h in ctx.hubs:
        callee = _callee(ctx, h)
        if callee not in _TRAIN_CALLEES:
            continue
        ctor = ctx.receiver_ctor(h)
        if callee == "fit_resample":
            continue
        if callee in ("fit_predict", "partial_fit") or _is_model(ctor) or (callee == "fit" and not _is_transformer(ctor)):
            out.append(h)
    return out

def _eval_hubs(ctx: _Ctx) -> list[str]:
    return [h for h in ctx.hubs if _callee(ctx, h) in _EVAL_CALLEES]

def _split_role_outputs(ctx: _Ctx, split: str) -> tuple[set[str], set[str]]:
    """Partition a split hub's outputs into (train_vars, test_vars) by name token, falling back to sklearn's positional convention (even=train, odd=test)."""
    outs = ctx.outputs(split)
    train, test = set(), set()
    for v in outs:
        toks = _tokens(ctx.g.nodes[v].get("label", v)) | _tokens(v)
        if toks & _TEST_TOKENS:
            test.add(v)
        elif toks & _TRAIN_TOKENS:
            train.add(v)
    if not train and not test and outs:
        for i, v in enumerate(outs):
            (train if i % 2 == 0 else test).add(v)
    return train, test

def _closure(g: nx.MultiDiGraph, seeds: set[str]) -> set[str]:
    """All nodes reachable downstream from seeds (inclusive)."""
    reach: set[str] = set(seeds)
    for s in seeds:
        reach |= nx.descendants(g, s)
    return reach

def _detect_preprocessing_before_split(ctx: _Ctx, splits: list[str]) -> list[LeakageFinding]:
    """D1: a transformer fit whose input feeds a later split ran on held-out rows."""
    findings: list[LeakageFinding] = []
    if not splits:
        return findings
    for p in _transformer_fit_hubs(ctx):
        for s in splits:
            if ctx.order.get(p, 0) >= ctx.order.get(s, 0):
                continue
            if any(nx.has_path(ctx.g, inp, s) for inp in ctx.inputs(p)):
                findings.append(LeakageFinding(
                    leakage_class=LeakageClass.PREPROCESSING_BEFORE_SPLIT,
                    node=p,
                    code=str(ctx.g.nodes[p].get("code", "")),
                    detail=f"'{_callee(ctx, p)}' is fit on data later partitioned by "
                           f"'{_callee(ctx, s)}' - statistics learned from held-out rows.",
                    source_node=s,
                ))
                break
    return findings

def _detect_target_leakage(ctx: _Ctx, splits: list[str], models: list[str]) -> list[LeakageFinding]:
    """D2: the target flows into the feature matrix of a model fit via a derivation."""
    findings: list[LeakageFinding] = []
    target_anchors: set[str] = set()
    for s in splits:
        ins = ctx.inputs(s)
        if len(ins) >= 2:
            target_anchors.add(ins[1])
    for m in models:
        ins = ctx.inputs(m)
        if len(ins) >= 2:
            target_anchors.add(ins[1])
    if not target_anchors:
        return findings

    split_set = set(splits)
    split_outputs: set[str] = set()
    for s in splits:
        split_outputs |= set(ctx.outputs(s))
    sub = ctx.g.subgraph([n for n in ctx.g.nodes() if n not in split_set])
    derived: set[str] = set()
    for a in target_anchors:
        if a not in sub:
            continue
        for succ in nx.descendants(sub, a):
            if ctx.g.nodes[succ].get(ATTR_NODE_TYPE) in COMPUTE_HUBS:
                derived |= set(ctx.outputs(succ))
    derived -= target_anchors
    derived -= split_outputs

    for m in models:
        ins = ctx.inputs(m)
        if not ins:
            continue
        feature = ins[0]
        feature_ancestors = nx.ancestors(ctx.g, feature) | {feature}
        hit = feature_ancestors & derived
        if hit:
            findings.append(LeakageFinding(
                leakage_class=LeakageClass.TARGET_LEAKAGE,
                node=m,
                code=str(ctx.g.nodes[m].get("code", "")),
                detail="feature matrix is derived from the target label "
                       f"(via {sorted(hit)[0]}).",
                source_node=sorted(hit)[0],
            ))
    return findings

def _detect_test_in_train(ctx: _Ctx, splits: list[str], models: list[str]) -> list[LeakageFinding]:
    """D3: held-out data reaches a training op (fixes the predict/score false positive)."""
    findings: list[LeakageFinding] = []
    test_seeds: set[str] = set()
    for s in splits:
        _, test_vars = _split_role_outputs(ctx, s)
        test_seeds |= test_vars
    if not test_seeds:
        test_seeds = {n for n, d in ctx.g.nodes(data=True)
                      if d.get(ATTR_NODE_TYPE) not in COMPUTE_HUBS and _tokens(n) & _TEST_TOKENS}
    if not test_seeds:
        return findings
    test_closure = _closure(ctx.g, test_seeds)
    for m in models:
        leaking = [inp for inp in ctx.inputs(m) if inp in test_closure]
        if leaking:
            findings.append(LeakageFinding(
                leakage_class=LeakageClass.TEST_IN_TRAIN,
                node=m,
                code=str(ctx.g.nodes[m].get("code", "")),
                detail=f"held-out data ({leaking[0]}) is fed to training op '{_callee(ctx, m)}'.",
                source_node=leaking[0],
            ))
    return findings

def _detect_metric_on_train(ctx: _Ctx, splits: list[str], models: list[str]) -> list[LeakageFinding]:
    """D4: an evaluation metric computed only on training data."""
    findings: list[LeakageFinding] = []
    evals = _eval_hubs(ctx)
    if not evals:
        return findings
    train_seeds: set[str] = set()
    for s in splits:
        train_vars, _ = _split_role_outputs(ctx, s)
        train_seeds |= train_vars
    for m in models:
        train_seeds |= set(ctx.inputs(m))
    if not train_seeds:
        return findings
    train_closure = _closure(ctx.g, train_seeds)
    test_seeds: set[str] = set()
    for s in splits:
        _, test_vars = _split_role_outputs(ctx, s)
        test_seeds |= test_vars
    test_closure = _closure(ctx.g, test_seeds)

    for e in evals:
        ins = set(ctx.inputs(e))
        if not ins:
            continue
        if ins & train_closure and not (ins & test_closure):
            findings.append(LeakageFinding(
                leakage_class=LeakageClass.METRIC_ON_TRAIN,
                node=e,
                code=str(ctx.g.nodes[e].get("code", "")),
                detail=f"metric '{_callee(ctx, e)}' is computed on training data "
                       "(no held-out inputs).",
            ))
    return findings

def _detect_no_holdout(ctx: _Ctx, splits: list[str], models: list[str]) -> list[LeakageFinding]:
    """D5: a model is trained but the notebook never carves off a test set."""
    if models and not splits:
        m = models[0]
        return [LeakageFinding(
            leakage_class=LeakageClass.NO_HOLDOUT_EVALUATION,
            node=m,
            code=str(ctx.g.nodes[m].get("code", "")),
            detail="model is fit but the notebook performs no train/test split.",
        )]
    return []

def analyze_leakage(g: nx.MultiDiGraph) -> list[LeakageFinding]:
    """Run all static leakage/anti-pattern detectors over a bipartite DFG."""
    ctx = _build_ctx(g)
    splits = _split_hubs(ctx)
    models = _model_fit_hubs(ctx)

    findings: list[LeakageFinding] = []
    findings += _detect_preprocessing_before_split(ctx, splits)
    findings += _detect_target_leakage(ctx, splits, models)
    findings += _detect_test_in_train(ctx, splits, models)
    findings += _detect_metric_on_train(ctx, splits, models)
    findings += _detect_no_holdout(ctx, splits, models)

    seen: set[tuple[str, str]] = set()
    unique: list[LeakageFinding] = []
    for f in findings:
        key = (f.leakage_class, f.node)
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique
