import json
import logging
from collections.abc import Iterator
from typing import Any

import nbformat

from SystemX.mining.protocols import NotebookIterator

from .client import GitHubClient

logger = logging.getLogger(__name__)

class GitHubNotebookIterator(NotebookIterator):
    """Streams notebooks from GitHub to be consumed by the MiningOrchestrator."""

    def __init__(self, client: GitHubClient, json_registry_path: str, start_index: int = 0) -> None:
        self.client = client
        self.current_index = start_index
        self.notebook_targets: list[dict[str, str]] = self._load_registry(json_registry_path)

    def _load_registry(self, path: str) -> list[dict[str, str]]:
        """Flattens the target registry into a list of fetchable notebook definitions."""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load registry {path}: {e}")
            return []

        targets = []
        for repo_key, contents in data.items():
            notebooks = contents.get("notebook_files", {})
            for name, meta in notebooks.items():
                targets.append({"filename": f"{repo_key.replace('/', '_')}_{name}", "git_url": meta["git_url"]})
        return targets

    def __iter__(self) -> Iterator[tuple[str, dict[str, Any]]]:
        return self

    def __next__(self) -> tuple[str, dict[str, Any]]:
        while self.current_index < len(self.notebook_targets):
            target = self.notebook_targets[self.current_index]
            self.current_index += 1

            notebook = self._fetch_notebook(target["git_url"])
            if notebook is not None:
                return target["filename"], notebook

        raise StopIteration

    def _fetch_notebook(self, git_url: str) -> dict[str, Any] | None:
        try:
            raw_content = self.client.get_decoded_file(git_url)
            return nbformat.reads(raw_content, as_version=4)
        except nbformat.reader.NotJSONError:
            logger.warning(f"File at {git_url} is not a valid Jupyter Notebook format.")
            return None
        except Exception as e:
            logger.error(f"Failed to process notebook {git_url}: {e}")
            return None
