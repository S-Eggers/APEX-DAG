import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import networkx as nx
from tqdm import tqdm

from ApexDAG.label_notebooks.labeler import ApexGraphLabeler
from ApexDAG.label_notebooks.token_policy import TokenBudgetPolicy
from ApexDAG.label_notebooks.utils import Config, load_config
from ApexDAG.llm.gemini_provider import GeminiProvider
from ApexDAG.llm.llm_provider import StructuredLLMProvider
from ApexDAG.sca.constants import DOMAIN_EDGE_TYPES
from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


def get_provider(config: Config) -> StructuredLLMProvider:
    """Factory to instantiate the correct LLM provider."""
    provider_type = getattr(config, "llm_provider", "google").lower()
    if provider_type == "google":
        return GeminiProvider(model_name=config.model_name, api_key=os.getenv("GEMINI_API_KEY"))

    raise ValueError(f"Unsupported LLM provider: {provider_type}")


def extract_code_from_ipynb(ipynb_path: Path) -> str:
    """Extracts raw python script from a Jupyter Notebook."""
    with open(ipynb_path, encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = ["".join(cell.get("source", [])) for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
    return "\n\n".join(code_cells)


def load_cytoscape_json(json_path: Path) -> tuple[list[dict[str, Any]], nx.MultiDiGraph]:
    """Parses Cytoscape JSON into a NetworkX graph for processing."""
    with open(json_path, encoding="utf-8") as f:
        elements = json.load(f)

    dfg = nx.MultiDiGraph()
    for el in elements:
        group = el.get("group")
        data = el.get("data", {})
        if group == "nodes":
            dfg.add_node(data.get("id"), **data)
        elif group == "edges":
            src, tgt = data.get("source"), data.get("target")
            edge_id = data.get("id", f"edge_{src}_{tgt}")
            dfg.add_edge(src, tgt, key=edge_id, **data)

    return elements, dfg


def sync_labels_to_json(elements: list[dict[str, Any]], labeled_dfg: nx.MultiDiGraph) -> list[dict[str, Any]]:
    """Updates the original JSON elements with new LLM-generated labels."""
    for el in elements:
        if el.get("group") == "edges":
            data = el["data"]
            src, tgt = data["source"], data["target"]
            edge_id = data.get("id", f"edge_{src}_{tgt}")

            if labeled_dfg.has_edge(src, tgt, key=edge_id):
                edge_data = labeled_dfg.edges[src, tgt, edge_id]
                label = edge_data.get("domain_label", "NOT_RELEVANT")

                data["domain_label"] = label
                data["reasoning"] = edge_data.get("reasoning", "")
                data["predicted_label"] = DOMAIN_EDGE_TYPES.get(label, 5)

    return elements


def main() -> None:
    parser = argparse.ArgumentParser(description="ApexDAG Batch Labeling Engine")
    parser.add_argument("--config", type=str, default="ApexDAG/label_notebooks/config.yaml")
    parser.add_argument("--input_dir", type=str, default="v2_labeling_workspace/annotations")
    parser.add_argument("--code_dir", type=str, default="v2_labeling_workspace/raw_dataset")
    parser.add_argument("--output_dir", type=str, default="v2_labeling_workspace/auto_labelled")
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    provider = get_provider(config)
    global_budget = TokenBudgetPolicy(max_tokens=config.max_tokens)

    json_files = list(Path(args.input_dir).glob("*.json"))
    processed = {f.name for f in output_path.glob("*.json")}
    pending = [f for f in json_files if f.name not in processed]

    logger.info(f"Starting batch: {len(pending)} files. Target: {config.model_name}")

    for json_file in tqdm(pending, desc="Processing Notebooks"):
        if global_budget.stop_event.is_set():
            logger.error("Global token budget exhausted. Terminating batch.")
            break

        ipynb_file = Path(args.code_dir) / json_file.name.replace(".json", ".ipynb")
        if not ipynb_file.exists():
            logger.warning(f"Source code missing for {json_file.name}. Skipping.")
            continue

        try:
            raw_code = extract_code_from_ipynb(ipynb_file)
            elements, G = load_cytoscape_json(json_file)

            labeler = ApexGraphLabeler(config, G, raw_code, provider, global_budget)
            labeled_dfg, _ = labeler.label_graph()

            updated_elements = sync_labels_to_json(elements, labeled_dfg)

            with open(output_path / json_file.name, "w", encoding="utf-8") as f:
                json.dump(updated_elements, f, indent=2)

        except Exception as e:
            logger.error(f"Critical failure on {json_file.name}: {e}")

    logger.info(f"Batch complete. Total tokens used: {global_budget.total_used}")


if __name__ == "__main__":
    main()
