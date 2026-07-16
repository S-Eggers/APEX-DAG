from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from SystemX.mining.github.client import GitHubClient
from SystemX.mining.github.code_search import GitHubCodeSearcher
from SystemX.mining.github.library_specs import LibrarySpec, resolve_spec
from SystemX.util.logger import configure_systemx_logger

configure_systemx_logger()
logger = logging.getLogger(__name__)

def registry_filename(library: str) -> str:
    """Filesystem-safe registry name for a library (apache-beam -> apache_beam)."""
    return library.strip().lower().replace("-", "_").replace("/", "_")

def build_library_registry(
    client: GitHubClient,
    spec: LibrarySpec,
    target: int,
    output_dir: Path,
    candidate_multiplier: int = 3,
    per_repo_cap: int = 2,
) -> Path:
    """Search for spec's notebooks and write a registry JSON."""
    limit = max(target * candidate_multiplier, target)
    searcher = GitHubCodeSearcher(client=client, per_repo_cap=per_repo_cap)

    registry: dict[str, dict[str, dict]] = {}
    count = 0
    for candidate in searcher.find_notebooks(spec, limit=limit):
        repo_key = candidate["repo_key"]
        entry = registry.setdefault(repo_key, {"notebook_files": {}, "data_files": {}})
        entry["notebook_files"][candidate["path"]] = {
            "git_url": candidate["git_url"],
            "html_url": candidate["html_url"],
        }
        count += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{registry_filename(spec.name)}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4)

    logger.info(
        "Registry for '%s': %d candidate notebooks across %d repos -> %s",
        spec.name, count, len(registry), output_path,
    )
    return output_path

def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build per-library notebook registries via GitHub Code Search.")
    parser.add_argument("--libraries", nargs="+", required=True, help="Library names, e.g. polars apache-beam")
    parser.add_argument("--target", type=int, default=100, help="Target kept notebooks per library (drives candidate over-fetch)")
    parser.add_argument("--candidate-multiplier", type=int, default=3, help="Fetch target x this many candidates per library")
    parser.add_argument("--per-repo-cap", type=int, default=2, help="Max notebooks taken from any single repository")
    parser.add_argument("--output-dir", type=str, default="data/library_registries", help="Directory for the per-library registry JSONs")
    args = parser.parse_args()

    token = os.getenv("GITHUB_TOKEN")
    if not token:
        logger.error("CRITICAL: GITHUB_TOKEN environment variable is missing. Terminating.")
        sys.exit(1)

    client = GitHubClient(token=token)
    output_dir = Path(args.output_dir)
    for library in args.libraries:
        spec = resolve_spec(library)
        build_library_registry(
            client=client,
            spec=spec,
            target=args.target,
            output_dir=output_dir,
            candidate_multiplier=args.candidate_multiplier,
            per_repo_cap=args.per_repo_cap,
        )

if __name__ == "__main__":
    main()
