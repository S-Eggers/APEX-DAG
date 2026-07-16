from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from .client import GitHubClient
from .library_specs import LibrarySpec

logger = logging.getLogger(__name__)

_MAX_PAGES = 10
_PER_PAGE = 100

class GitHubCodeSearcher:
    """Yields candidate notebooks for a library, deduped and repo-capped."""

    def __init__(self, client: GitHubClient, per_repo_cap: int = 2) -> None:
        self.client = client
        self.per_repo_cap = per_repo_cap

    def find_notebooks(self, spec: LibrarySpec, limit: int) -> Iterator[dict[str, str]]:
        """Yield up to limit candidate {repo_key, path, git_url, html_url} dicts."""
        seen: set[tuple[str, str]] = set()
        repo_counts: dict[str, int] = defaultdict(int)
        yielded = 0

        for keyword in spec.search:
            query = f"{keyword} in:file extension:ipynb"
            logger.info("Code search for %s: q=%r", spec.name, query)

            for page in range(1, _MAX_PAGES + 1):
                if yielded >= limit:
                    return

                data = self.client.search_code(query, page=page, per_page=_PER_PAGE)
                items = data.get("items", []) if isinstance(data, dict) else []
                if not items:
                    break

                for item in items:
                    candidate = self._to_candidate(item)
                    if candidate is None:
                        continue

                    repo_key = candidate["repo_key"]
                    key = (repo_key, candidate["path"])
                    if key in seen:
                        continue
                    if repo_counts[repo_key] >= self.per_repo_cap:
                        continue

                    seen.add(key)
                    repo_counts[repo_key] += 1
                    yielded += 1
                    yield candidate

                    if yielded >= limit:
                        return

                if len(items) < _PER_PAGE:
                    break

        logger.info("Code search for %s yielded %d candidates.", spec.name, yielded)

    @staticmethod
    def _to_candidate(item: dict[str, Any]) -> dict[str, str] | None:
        repo = item.get("repository", {})
        repo_key = repo.get("full_name")
        path = item.get("path")
        git_url = item.get("git_url")
        if not (repo_key and path and git_url):
            return None
        return {
            "repo_key": repo_key,
            "path": path,
            "git_url": git_url,
            "html_url": item.get("html_url", ""),
        }
