import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

from SystemX.execution.heuristic_predictor import HeuristicOrderPredictor
from SystemX.execution.learned_predictor import LearnedOrderPredictor
from SystemX.nn.data.v2.execution_order_dataset import (
    CellPairFeaturizer,
    _cells_with_counts,
    build_notebook_pairs,
    build_permuted_notebook_pairs,
    is_linear,
    is_valid_supervision,
)
from SystemX.parser.graph_parser import GraphParser
from SystemX.sca.cell_graph_projector import CellGraphProjector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("SystemX.experiment.execution_order.evaluate")

def _fit_mlp(x: np.ndarray, y: np.ndarray, w: np.ndarray, epochs: int, pretrain: tuple | None = None, pretrain_epochs: int = 0):
    import torch
    import torch.nn as nn

    from SystemX.nn.models.v2.mlp import ComputeHubMLP

    model = ComputeHubMLP(in_features=x.shape[1], num_classes=2)
    criterion = nn.CrossEntropyLoss(reduction="none")

    def run(features: np.ndarray, labels: np.ndarray, weights: np.ndarray, n_epochs: int, lr: float) -> None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        x_t, y_t = torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
        w_t = torch.tensor(weights / weights.mean(), dtype=torch.float32)
        model.train()
        for _ in range(n_epochs):
            optimizer.zero_grad()
            loss = (criterion(model(x_t), y_t) * w_t).mean()
            loss.backward()
            optimizer.step()

    if pretrain is not None and pretrain_epochs > 0:
        run(*pretrain, pretrain_epochs, 1e-3)
        run(x, y, w, epochs, 3e-4)
    else:
        run(x, y, w, epochs, 1e-3)
    model.eval()

    def scorer(features: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(model(torch.tensor(features, dtype=torch.float32)), dim=1)[:, 1].numpy()

    return scorer

def _fit_xgboost(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    import xgboost as xgb

    model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, objective="binary:logistic", eval_metric="logloss", n_jobs=-1)
    model.fit(x, y.astype(int), sample_weight=w)
    return lambda features: model.predict_proba(features)[:, 1]

def _evaluate_order(predicted: list[str], true_rank: dict[str, int], cg) -> dict[str, float]:
    rank = {cell: i for i, cell in enumerate(predicted)}
    cells = [c for c in predicted if c in true_rank]
    metrics: dict[str, float] = {}

    if len(cells) >= 2:
        tau, _ = kendalltau([rank[c] for c in cells], [true_rank[c] for c in cells])
        metrics["kendall_tau"] = float(tau) if tau == tau else 0.0

    def pair_acc(pairs: list[tuple[str, str]]) -> float | None:
        scored = [(a, b) for a, b in pairs if a in rank and b in rank and a in true_rank and b in true_rank and true_rank[a] != true_rank[b]]
        if not scored:
            return None
        hits = sum(1 for a, b in scored if (rank[a] < rank[b]) == (true_rank[a] < true_rank[b]))
        return hits / len(scored)

    all_pairs = [(cells[i], cells[j]) for i in range(len(cells)) for j in range(i + 1, len(cells))]
    dep_pairs = [(e.src_cell, e.dst_cell) for e in cg.edges()]
    ambig_pairs = [(c, e.dst_cell) for e in cg.ambiguities() for c in e.candidate_def_cells]

    for name, pairs in (("pairwise_acc", all_pairs), ("pairwise_acc_dep", dep_pairs), ("pairwise_acc_ambig", ambig_pairs)):
        acc = pair_acc(pairs)
        if acc is not None:
            metrics[name] = acc

    forward_edges = [(e.src_cell, e.dst_cell) for e in cg.edges() if not e.ambiguous and not e.out_of_order]
    if forward_edges:
        metrics["topo_validity"] = sum(1 for a, b in forward_edges if rank.get(a, 0) < rank.get(b, 0)) / len(forward_edges)
    return metrics

def evaluate_fold(test_notebooks: list[tuple[str, list[dict]]], predictors: dict, featurizer: CellPairFeaturizer, seed: int) -> dict:
    parser = GraphParser()
    rng = random.Random(seed)
    per_metric: dict[str, list[float]] = defaultdict(list)

    for _name, cells in test_notebooks:
        try:
            dfg = parser.parse(cells)
        except Exception:
            continue
        counts = {c["cell_id"]: c["execution_count"] for c in cells}
        true_order = sorted(counts, key=counts.get)
        true_rank = {cell: i for i, cell in enumerate(true_order)}
        nonlinear = not is_linear(cells)

        for pred_name, predictor in predictors.items():
            cg = CellGraphProjector(dfg, cells).project()
            if predictor == "document":
                order = cg.cells
            elif predictor == "random":
                order = list(cg.cells)
                rng.shuffle(order)
            else:
                report = predictor.predict(cg, dfg=dfg)
                order = report.predicted_order

            for metric, value in _evaluate_order(order, true_rank, cg).items():
                per_metric[f"{pred_name}/{metric}"].append(value)
                if nonlinear:
                    per_metric[f"{pred_name}/{metric}_nonlinear"].append(value)

    return {k: (float(np.mean(v)), len(v)) for k, v in per_metric.items()}

def main() -> None:
    parser = argparse.ArgumentParser(description="K-fold evaluation of execution-order prediction")
    parser.add_argument("--notebooks_dir", default="data/jetbrains_dataset/notebooks")
    parser.add_argument("--limit", type=int, default=None, help="Max valid notebooks to use")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100, help="MLP epochs per fold")
    parser.add_argument("--features", default="struct", choices=["struct", "standard"])
    parser.add_argument("--models", nargs="*", default=["mlp", "xgboost"], choices=["mlp", "xgboost"],
                        help="Which learned predictors to fit (default both). Use 'xgboost' alone to skip the slow MLP.")
    parser.add_argument("--nonlinear_weight", type=float, default=2.0)
    parser.add_argument("--augment_permutations", type=int, default=0, help="Self-supervision: permutations per train-fold notebook mixed into training (0 = count supervision only)")
    parser.add_argument("--permutation_weight", type=float, default=1.0)
    parser.add_argument("--pretrain_epochs", type=int, default=0, help="MLP only: pretrain on permutation pairs, then fine-tune on count pairs")
    parser.add_argument("--random_fraction", type=float, default=0.3, help="Fraction of unconstrained shuffles among sampled permutations")
    parser.add_argument("--output", default="output/results/execution_order_cv.json")
    args = parser.parse_args()

    if args.features == "struct":
        featurizer = CellPairFeaturizer(None)
    else:
        from SystemX.nn.data.v2.fasttext_embedding import FastTextEmbeddingV2
        from SystemX.nn.data.v2.feature_extractor import ComputeHubFeatureExtractor, FeatureGroup

        featurizer = CellPairFeaturizer(ComputeHubFeatureExtractor(FastTextEmbeddingV2(), groups=FeatureGroup.STANDARD & ~FeatureGroup.CELL_POS))

    paths = sorted(Path(args.notebooks_dir).glob("*.ipynb"))
    rng = random.Random(args.seed)
    rng.shuffle(paths)

    notebooks: list[tuple[str, list[dict]]] = []
    for path in paths:
        cells = _cells_with_counts(path)
        if cells and is_valid_supervision(cells):
            notebooks.append((path.name, cells))
        if args.limit and len(notebooks) >= args.limit:
            break
    nonlinear_count = sum(1 for _, cells in notebooks if not is_linear(cells))
    logger.info("Using %d valid notebooks (%d nonlinear).", len(notebooks), nonlinear_count)
    if len(notebooks) < args.folds:
        logger.error("Not enough valid notebooks for %d folds.", args.folds)
        return

    fold_results: list[dict] = []
    for fold in range(args.folds):
        test = [nb for i, nb in enumerate(notebooks) if i % args.folds == fold]
        train = [nb for i, nb in enumerate(notebooks) if i % args.folds != fold]

        gp = GraphParser()
        xs, ys, ws = [], [], []
        for _name, cells in train:
            built = build_notebook_pairs(cells, featurizer, gp)
            if built is None:
                continue
            x, y, nonlinear = built
            xs.append(x)
            ys.append(y)
            ws.append(np.full(len(y), args.nonlinear_weight if nonlinear else 1.0, dtype=np.float32))
        x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
        logger.info("Fold %d: %d train notebooks -> %d pairs, %d test notebooks.", fold, len(train), len(y), len(test))

        pretrain = None
        if args.augment_permutations > 0 or args.pretrain_epochs > 0:
            pxs, pys = [], []
            for i, (_name, cells) in enumerate(train):
                built = build_permuted_notebook_pairs(cells, featurizer, gp, n_permutations=args.augment_permutations or 3, random_fraction=args.random_fraction, seed=args.seed + fold * 10000 + i)
                if built is None:
                    continue
                pxs.append(built[0])
                pys.append(built[1])
            if pxs:
                px, py = np.concatenate(pxs), np.concatenate(pys)
                pw = np.full(len(py), args.permutation_weight, dtype=np.float32)
                if args.pretrain_epochs > 0:
                    pretrain = (px, py, pw)
                    logger.info("Fold %d: pretraining on %d permutation pairs.", fold, len(py))
                else:
                    x, y, w = np.concatenate([x, px]), np.concatenate([y, py]), np.concatenate([w, pw])
                    logger.info("Fold %d: mixed in %d permutation pairs -> %d total.", fold, len(py), len(y))

        xgb_train = (np.concatenate([x, pretrain[0]]), np.concatenate([y, pretrain[1]]), np.concatenate([w, pretrain[2]])) if pretrain is not None else (x, y, w)
        predictors = {
            "document": "document",
            "random": "random",
            "heuristic": HeuristicOrderPredictor(),
        }
        if "mlp" in args.models:
            predictors["mlp"] = LearnedOrderPredictor(
                _fit_mlp(x, y, w, args.epochs, pretrain=pretrain, pretrain_epochs=args.pretrain_epochs), featurizer, name="mlp")
        if "xgboost" in args.models:
            predictors["xgboost"] = LearnedOrderPredictor(_fit_xgboost(*xgb_train), featurizer, name="xgboost")
        result = evaluate_fold(test, predictors, featurizer, seed=args.seed + fold)
        fold_results.append(result)

        for constrained in ("heuristic", *args.models):
            key = f"{constrained}/topo_validity"
            if key in result and result[key][0] < 1.0:
                logger.warning("%s violated topological validity: %.4f", constrained, result[key][0])

    aggregated: dict[str, dict] = {}
    keys = sorted({k for fr in fold_results for k in fr})
    for key in keys:
        values = [fr[key][0] for fr in fold_results if key in fr]
        counts = sum(fr[key][1] for fr in fold_results if key in fr)
        aggregated[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": counts}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": vars(args), "folds": len(fold_results), "notebooks": len(notebooks), "nonlinear": nonlinear_count, "metrics": aggregated}, indent=2))
    logger.info("Results written to %s", out_path)

    logger.info("%-34s %8s %8s %6s", "metric", "mean", "std", "n")
    for key, stats in aggregated.items():
        logger.info("%-34s %8.4f %8.4f %6d", key, stats["mean"], stats["std"], stats["n"])

if __name__ == "__main__":
    main()
