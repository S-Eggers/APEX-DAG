import base64
import logging
import time
from typing import Any
from urllib.parse import quote_plus

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_SEARCH_MIN_INTERVAL_SEC = 6.0

class GitHubClient:
    """Handles authenticated GitHub API requests and rate limit enforcement."""

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("GitHub token must be provided.")

        self.headers: dict[str, str] = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

        self.session = requests.Session()
        self.session.headers.update(self.headers)

        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        self._last_search_ts = 0.0

    def _handle_rate_limit(self, response: requests.Response) -> None:
        """Blocks execution if the rate limit is exhausted."""
        remaining = int(response.headers.get("X-RateLimit-Remaining", 1))
        if remaining > 0 and response.status_code != 403:
            return

        reset_timestamp = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
        sleep_duration = max(0, reset_timestamp - int(time.time())) + 1

        logger.warning(f"GitHub API rate limit exhausted. Sleeping for {sleep_duration} seconds.")
        time.sleep(sleep_duration)

    def get_json(self, url: str) -> dict[str, Any] | list[dict[str, Any]]:
        """Executes a GET request and returns JSON, respecting rate limits."""
        while True:
            try:
                response = self.session.get(url, timeout=15)
                self._handle_rate_limit(response)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error fetching {url}: {e}")
                raise

    def search_code(self, query: str, page: int = 1, per_page: int = 100) -> dict[str, Any]:
        """Runs a Code Search API query for file contents, throttled to the search budget."""
        elapsed = time.time() - self._last_search_ts
        if elapsed < _SEARCH_MIN_INTERVAL_SEC:
            time.sleep(_SEARCH_MIN_INTERVAL_SEC - elapsed)

        url = f"https://api.github.com/search/code?q={quote_plus(query)}&per_page={per_page}&page={page}"
        try:
            data = self.get_json(url)
        finally:
            self._last_search_ts = time.time()

        return data if isinstance(data, dict) else {"total_count": 0, "items": []}

    def get_decoded_file(self, url: str) -> str:
        """Fetches and decodes a base64 encoded file from the GitHub API."""
        data = self.get_json(url)
        if isinstance(data, dict) and data.get("encoding") == "base64":
            return base64.b64decode(data.get("content", "")).decode("utf-8")
        raise ValueError(f"Expected base64 encoded payload from {url}")

    def get_repository_tree(self, owner: str, repo: str, default_branch: str) -> list[dict[str, Any]]:
        """Fetches the entire repository file tree in a single API call using the Git Trees API."""
        url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{default_branch}?recursive=1"
        try:
            data = self.get_json(url)
            if isinstance(data, dict) and "tree" in data:
                return data["tree"]
            return []
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code in (404, 409, 451):
                logger.warning(f"Could not fetch tree for {owner}/{repo}: HTTP {e.response.status_code}")
                return []
            raise
