import logging
from collections.abc import Iterator
from typing import Any

import pandas as pd

from .client import GitHubClient

logger = logging.getLogger(__name__)


class GitHubRepositoryScanner:
    """Discovers repositories using date-partitioned subqueries to bypass API limits."""

    def __init__(self, client: GitHubClient, query: str) -> None:
        self.client = client
        self.base_url = "https://api.github.com/search/repositories?q="
        self.query = query

    def _generate_subqueries(self, start_date: str, end_date: str) -> list[str]:
        """Partitions the date range into monthly windows."""
        dates = pd.date_range(start_date, end_date, freq="ME").strftime("%Y-%m-%d").tolist()
        subqueries = []
        prefix = "+" if self.query else ""

        for date in dates:
            subqueries.append(f"{prefix}language:jupyter-notebook+pushed:<={date}&per_page=100&sort=updated&order=desc")
            subqueries.append(f"{prefix}language:jupyter-notebook+pushed:>={date}&per_page=100&sort=updated&order=asc")

        return subqueries

    def scan(self, start_date: str, end_date: str) -> Iterator[dict[str, Any]]:
        """Yields repository metadata items."""
        subqueries = self._generate_subqueries(start_date, end_date)
        visited = set()

        for subquery in subqueries:
            url = f"{self.base_url}{self.query}{subquery}"
            data = self.client.get_json(url)

            if not isinstance(data, dict):
                continue

            total_count = data.get("total_count", 0)
            pages = max(1, min(total_count, 1000) // 100)

            for page in range(1, pages + 1):
                page_url = f"{url}&page={page}"
                page_data = self.client.get_json(page_url)

                if isinstance(page_data, dict) and "items" in page_data:
                    for item in page_data["items"]:
                        repo_id = (item["owner"]["login"], item["name"])
                        if repo_id not in visited:
                            visited.add(repo_id)
                            yield item
