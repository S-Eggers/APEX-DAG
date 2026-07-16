import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv
from tqdm import tqdm

from SystemX.mining.github.client import GitHubClient
from SystemX.mining.github.scanner import GitHubRepositoryScanner

logger = logging.getLogger(__name__)


def generate_registry(client: GitHubClient, query: str, start_date: str, end_date: str, output_path: str) -> None:
    """Scans for repositories and builds a registry of potential notebook targets."""
    scanner = GitHubRepositoryScanner(client=client, query=query)
    registry: dict = {}

    data_extensions = (".csv", ".json", ".xls", ".xlsx", ".parquet", ".sql", ".yml", ".yaml", ".zip", ".tar.gz", ".7z", ".xml")

    logger.info(f"Initiating repository scan from {start_date} to {end_date} for query: '{query}'")

    try:
        for repo_item in tqdm(scanner.scan(start_date, end_date), desc="Scanning Repositories"):
            owner = repo_item["owner"]["login"]
            repo_name = repo_item["name"]
            default_branch = repo_item.get("default_branch", "main")
            repo_key = f"{owner}/{repo_name}"

            tree = client.get_repository_tree(owner, repo_name, default_branch)

            notebooks = {}
            data_files = {}

            for item in tree:
                if item.get("type") == "blob":
                    path = item.get("path", "")

                    if "site-packages" in path or ".venv" in path:
                        continue

                    file_meta = {"git_url": item.get("url"), "html_url": f"https://github.com/{repo_key}/blob/{default_branch}/{path}"}

                    if path.endswith(".ipynb"):
                        notebooks[path] = file_meta
                    elif path.endswith(data_extensions):
                        data_files[path] = file_meta

            if notebooks:
                registry[repo_key] = {"notebook_files": notebooks, "data_files": data_files}

    except KeyboardInterrupt:
        logger.warning("Scan interrupted by user. Saving partial registry.")
    except Exception as e:
        logger.exception(f"Scan failed: {e}. Saving partial registry.")
    finally:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=4)
        logger.info(f"Registry saved to {output_path} with {len(registry)} valid repositories.")


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="SystemX GitHub Repository Scanner")
    parser.add_argument("--query", type=str, default="machine learning", help="Search query")
    parser.add_argument("--start", type=str, default="2025-12-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2026-05-14", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="./data/github_registry.json", help="Output JSON path")
    args = parser.parse_args()

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.error("CRITICAL: GITHUB_TOKEN environment variable is missing. Terminating.")
        sys.exit(1)

    client = GitHubClient(token=github_token)
    generate_registry(client, args.query, args.start, args.end, args.output)
