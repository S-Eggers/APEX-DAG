import argparse
import json
import logging
from pathlib import Path

import networkx as nx
from tqdm import tqdm

from ApexDAG.label_notebooks.labeler import ApexGraphLabeler
from ApexDAG.label_notebooks.utils import load_config
from ApexDAG.sca.constants import DOMAIN_EDGE_TYPES
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


def extract_code_from_ipynb(ipynb_path: Path) -> str:
    """Extracts a raw python script from a Jupyter Notebook file."""
    with open(ipynb_path, encoding="utf-8") as f:
        nb = json.load(f)

    code_lines = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            code_lines.append(source)

    return "\n\n".join(code_lines)


def load_cytoscape_json(json_path: Path) -> tuple[list, nx.MultiDiGraph]:
    """Loads Cytoscape JSON, returning the raw dict and a NetworkX MultiDiGraph."""
    with open(json_path, encoding="utf-8") as f:
        elements = json.load(f)

    dfg = nx.MultiDiGraph()
    for el in elements:
        group = el.get("group")
        data = el.get("data", {})
        if group == "nodes":
            dfg.add_node(data.get("id"), **data)
        elif group == "edges":
            src = data.get("source")
            tgt = data.get("target")
            key = data.get("id", f"edge_{src}_{tgt}")
            dfg.add_edge(src, tgt, key=key, **data)

    return elements, dfg


def sync_labels_to_json(elements: list, labeled_dfg: nx.MultiDiGraph) -> list:
    """Transfers the LLM labels back to the UI JSON payload securely."""
    for el in elements:
        if el.get("group") == "edges":
            src = el["data"]["source"]
            tgt = el["data"]["target"]
            key = el["data"].get("id", f"edge_{src}_{tgt}")

            if labeled_dfg.has_edge(src, tgt, key=key):
                edge_data = labeled_dfg.edges[src, tgt, key]

                domain_label = edge_data.get("domain_label", "NOT_RELEVANT")
                el["data"]["domain_label"] = domain_label
                el["data"]["reasoning"] = edge_data.get("reasoning", "")

                # Re-sync to the integer enum for ML training later
                el["data"]["predicted_label"] = DOMAIN_EDGE_TYPES.get(domain_label, 5)

    return elements


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, default="ApexDAG/label_notebooks/config.yaml"
    )
    parser.add_argument(
        "--json_dir", type=str, default="v2_labeling_workspace/annotations"
    )
    parser.add_argument(
        "--ipynb_dir", type=str, default="v2_labeling_workspace/raw_dataset"
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="v2_labeling_workspace/auto_labelled_annotations",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    target_path = Path(args.target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    json_files = list(Path(args.json_dir).glob("*.json"))
    processed = set(f.name for f in target_path.glob("*.json"))

    pending_files = [f for f in json_files if f.name not in processed]
    budget = getattr(config, "max_tokens", "INF")
    logger.info(f"Found {len(pending_files)} pending JSONs. Token Budget: {budget}")

    for json_file in tqdm(pending_files, desc="Labeling Batch"):
        ipynb_file = Path(args.ipynb_dir) / json_file.name.replace(".json", ".ipynb")

        if not ipynb_file.exists():
            logger.error(f"Missing notebook code for {json_file.name}. Skipping.")
            continue

        try:
            raw_code = extract_code_from_ipynb(ipynb_file)
            elements, G = load_cytoscape_json(json_file)

            labeler = ApexGraphLabeler(config, G, raw_code)
            labeled_dfg, tokens_spent = labeler.label_graph()

            config.max_tokens -= tokens_spent
            message = f"""
Spent {tokens_spent} tokens on {json_file.name}.
Remaining Budget: {config.max_tokens}"
            """
            logger.info(message)

            updated_elements = sync_labels_to_json(elements, labeled_dfg)

            output_file = target_path / json_file.name
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(updated_elements, f, indent=2)

            if config.max_tokens <= 0:
                logger.error("GLOBAL BUDGET EXCEEDED. Safely terminating batch runner.")
                break

        except Exception as e:
            logger.error(f"Failed to process {json_file.name}: {e}", exc_info=True)
