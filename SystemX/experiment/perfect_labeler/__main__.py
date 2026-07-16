import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import nbformat
import torch
import yaml

from SystemX.nn.data.v1.encoder import GraphEncoder
from SystemX.nn.models.v1.gat import MultiTaskGATv1
from SystemX.pipeline.lineage_pipeline_factory import LineagePipelineFactory
from SystemX.util.logger import configure_systemx_logger
from SystemX.vamsa.experiment.golden_types import GoldenElementData, GoldenGraph

configure_systemx_logger()
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=SyntaxWarning, message=".*invalid.*")

def load_notebook_cells(ipynb_path: Path) -> list[dict]:
    """Extracts raw cell dictionaries."""
    with open(ipynb_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    return [dict(c) for c in nb.cells if c.cell_type == "code"]

def load_ml_assets(config_path: Path, checkpoint_path: Path) -> dict[str, object] | None:
    """Initializes GAT and Encoder."""
    if not config_path.exists() or not checkpoint_path.exists():
        logger.error("Missing required ML assets.")
        return None
    with open(config_path) as f:
        config = yaml.safe_load(f)
    graph_encoder = GraphEncoder(Path(config["encoded_checkpoint_path"]), min_nodes=3, min_edges=2, load_encoded_old_if_exist=False, mode="REVERSED")
    model = MultiTaskGATv1(
        hidden_dim=config["hidden_dim"],
        dim_embed=config["dim_embed"],
        num_heads=config["num_heads"],
        edge_classes=config["node_classes"],
        node_classes=config["edge_classes"],
        residual=config["residual"],
        dropout=config["dropout"],
        number_gat_blocks=config["number_gat_blocks"],
        task=["node_classification"],
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to("cpu")
    model.eval()
    return {"encoder": graph_encoder, "model": model}

def normalize_id(node_id: str) -> str:
    """Standardizes IDs to bridge the naming gap between Parser and Golden Set."""
    if not node_id:
        return ""
    return str(node_id).replace("legacy_f_", "fallback_").replace("legacy_fallback_", "fallback_")

def extract_edges_from_payload(payload: dict) -> set[tuple[str, str]]:
    """Extracts ground truth edges from the Golden JSON."""
    edges: set[tuple[str, str]] = set()
    for element in payload.get("elements", []):
        data: GoldenElementData = element.get("data", {})
        source, target = data.get("source"), data.get("target")
        if source is not None and target is not None:
            edges.add((str(source), str(target)))
    return edges

def main() -> None:
    parser = argparse.ArgumentParser(description="SystemX Structural Connectivity Audit")
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--annotations_dir", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    args = parser.parse_args()

    raw_dir, annotations_dir = Path(args.raw_dir), Path(args.annotations_dir)
    assets = load_ml_assets(Path(args.config_path), Path(args.checkpoint_path))
    if not assets:
        sys.exit(1)

    pipeline = LineagePipelineFactory.create(request_payload={"llmClassification": False, "highlightRelevantSubgraphs": False}, model=assets)

    global_tp, global_fp, global_fn = 0.0, 0.0, 0.0
    total_notebooks = 0

    print("\n" + "=" * 60)
    print(f"{'NOTEBOOK':<45} | {'GOLD':<5} | {'MATCH':<5}")
    print("-" * 60)

    for json_path in annotations_dir.glob("*.json"):
        notebook_name = json_path.name.replace(".json", ".ipynb")
        ipynb_path = raw_dir / notebook_name
        if not ipynb_path.exists():
            continue

        total_notebooks += 1
        with open(json_path, encoding="utf-8") as f:
            golden_json: GoldenGraph = json.load(f)

        golden_edges = extract_edges_from_payload(golden_json)
        source_cells = load_notebook_cells(ipynb_path)

        try:
            dfg = pipeline.parser.parse(source_cells)
            dfg.get_state().optimize()
            G = dfg.get_graph()

            dfg_edges = set()
            for u, v in G.edges():
                norm_u = normalize_id(u)
                norm_v = normalize_id(v)
                dfg_edges.add((norm_u, norm_v))

            tp = len(dfg_edges.intersection(golden_edges))
            fp = len(dfg_edges - golden_edges)
            fn = len(golden_edges - dfg_edges)

            print(f"{notebook_name:<45} | {len(golden_edges):<5} | {tp:<5}")

            global_tp += tp
            global_fp += fp
            global_fn += fn

        except Exception as e:
            if "already exists" in str(e):
                logger.error(f"Parser State Conflict: {notebook_name}")
            else:
                logger.error(f"Structural Error {notebook_name}: {e}")
            global_fn += len(golden_edges)
            continue

    precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("=" * 60)
    print("STRUCTURAL AUDIT RESULTS (Bypassing Serializer/ML)")
    print("-" * 60)
    print(f"Global Structural Precision: {precision:.4f}")
    print(f"Global Structural Recall:    {recall:.4f}")
    print(f"Global Structural F1-Score:  {f1:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
