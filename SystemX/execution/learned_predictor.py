import logging
from pathlib import Path

import numpy as np

from SystemX.execution.heuristic_predictor import HeuristicOrderPredictor
from SystemX.execution.types import ExecutionStateReport
from SystemX.nn.data.v2.execution_order_dataset import CellPairFeaturizer
from SystemX.sca.cell_graph_projector import CellDependencyGraph

logger = logging.getLogger(__name__)

class LearnedOrderPredictor:
    def __init__(self, scorer, featurizer: CellPairFeaturizer, name: str = "learned") -> None:
        """scorer(x: np.ndarray[n, pair_dim]) -> np.ndarray[n] returns P(first cell of the pair executed before the second)."""
        self.scorer = scorer
        self.featurizer = featurizer
        self.name = name
        self._backbone = HeuristicOrderPredictor()

    def predict(self, cg: CellDependencyGraph, dfg=None) -> ExecutionStateReport:
        cells = cg.cells
        if len(cells) < 2:
            return self._backbone.predict(cg)

        dfg_graph = dfg.get_graph() if (dfg is not None and self.featurizer.hub_extractor) else None
        prob = self._pairwise_matrix(cg, cells, dfg_graph)
        index = {cell: i for i, cell in enumerate(cells)}

        self._rebind_ambiguous_edges(cg, prob, index)

        priority = prob.sum(axis=1)
        tie_break = [cell for _, cell in sorted(((-priority[index[c]], index[c]), c) for c in cells)]

        report = self._backbone.predict(cg, tie_break=tie_break)
        report.notebook_flags["order_predictor"] = self.name
        return report

    def _pairwise_matrix(self, cg: CellDependencyGraph, cells: list[str], dfg_graph) -> np.ndarray:
        n = len(cells)
        pairs = [(cells[i], cells[j]) for i in range(n) for j in range(n) if i != j]
        cell_cache: dict = {}
        x = np.stack([self.featurizer.pair_vector(cg, a, b, dfg_graph, cell_cache) for a, b in pairs])
        scores = self.scorer(x)

        index = {cell: i for i, cell in enumerate(cells)}
        prob = np.zeros((n, n), dtype=np.float32)
        for (a, b), score in zip(pairs, scores, strict=True):
            prob[index[a], index[b]] = score
        return prob

    def _rebind_ambiguous_edges(self, cg: CellDependencyGraph, prob: np.ndarray, index: dict[str, int]) -> None:
        for u, v, key in list(cg.graph.edges(keys=True)):
            edge = cg.graph.edges[u, v, key]["edge"]
            if not edge.ambiguous or len(edge.candidate_def_cells) < 2:
                continue
            candidates = [c for c in edge.candidate_def_cells if c in index and c != edge.dst_cell]
            if len(candidates) < 2:
                continue

            def last_definer_score(c: str, _dst: str = edge.dst_cell, _cands: list[str] = candidates) -> float:
                before_reader = prob[index[c], index[_dst]]
                after_others = sum(prob[index[o], index[c]] for o in _cands if o != c)
                return before_reader + after_others

            best = max(candidates, key=last_definer_score)
            if best != edge.src_cell:
                logger.debug("Rebinding `%s` for %s: %s -> %s", edge.name, edge.dst_cell, edge.src_cell, best)
                cg.graph.remove_edge(u, v, key)
                edge.src_cell = best
                cg.graph.add_edge(best, edge.dst_cell, edge=edge)

def load_exec_order_predictor(manifest_key: str, path: Path, get_embedding=None) -> LearnedOrderPredictor | None:
    """Load a predictor from an exec_order_<family>_<preset> checkpoint."""
    try:
        family = "mlp" if "mlp" in manifest_key else "xgboost"
        if family == "mlp":
            import torch

            from SystemX.nn.models.v2.mlp import ComputeHubMLP

            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            meta = checkpoint.get("meta", {})
            model = ComputeHubMLP(in_features=meta["in_features"], hidden=meta.get("hidden", 256), num_classes=2)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            def scorer(x: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    logits = model(torch.tensor(x, dtype=torch.float32))
                    return torch.softmax(logits, dim=1)[:, 1].numpy()

            preset = meta.get("featurizer_preset", "struct")
        else:
            import xgboost as xgb

            booster = xgb.XGBClassifier()
            booster.load_model(str(path))

            def scorer(x: np.ndarray) -> np.ndarray:
                return booster.predict_proba(x)[:, 1]

            meta_path = path.with_suffix(".meta.json")
            preset = "struct"
            if meta_path.exists():
                import json

                preset = json.loads(meta_path.read_text()).get("featurizer_preset", "struct")

        hub_extractor = None
        if preset == "standard":
            if get_embedding is None:
                logger.warning("exec-order checkpoint %s needs embeddings but none available; skipping.", manifest_key)
                return None
            from SystemX.nn.data.v2.feature_extractor import ComputeHubFeatureExtractor, FeatureGroup

            groups = FeatureGroup.STANDARD & ~FeatureGroup.CELL_POS
            hub_extractor = ComputeHubFeatureExtractor(get_embedding(), groups=groups)

        featurizer = CellPairFeaturizer(hub_extractor)
        return LearnedOrderPredictor(scorer, featurizer, name=manifest_key)
    except Exception as exc:
        logger.error("Failed loading exec-order predictor %s from %s: %s", manifest_key, path, exc, exc_info=True)
        return None

def resolve_learned_order_predictor(backend: str, models: dict) -> LearnedOrderPredictor | None:
    """Resolve a frontend backend selection (e.g."""
    predictor = models.get(backend)
    if isinstance(predictor, LearnedOrderPredictor):
        return predictor
    for key, candidate in models.items():
        if key.startswith(backend) and isinstance(candidate, LearnedOrderPredictor):
            return candidate
    logger.warning("Execution-order backend %r not loaded; falling back to heuristic.", backend)
    return None
