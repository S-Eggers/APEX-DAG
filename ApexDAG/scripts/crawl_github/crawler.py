import requests
import time
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

class GitHubNotebookSearch:
    def __init__(self, token: str, query: str, per_page: int = 30, max_results: int = 10, results_file_path: str = 'data/raw/notebooks.json'):
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("app.log", mode='a')
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.GITHUB_TOKEN = token
        self.QUERY = query
        self.PER_PAGE = per_page
        self.MAX_RESULTS = max_results

        self.HEADERS = {
            "Authorization": f"token {self.GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        self.SEARCH_URL = "https://api.github.com/search/code"
        
        self.results_file_path = results_file_path

    def search_notebooks(self):
        """
        Search for Jupyter notebooks on GitHub.

        Returns:
            list: List of search results with name, path, and repository info.
        """
        results = []
        page = 1

        while len(results) < self.MAX_RESULTS:
            self.logger.info(f"({len(results)}/{self.MAX_RESULTS}) Fetching page {page}...")
            response = requests.get(
                self.SEARCH_URL,
                headers=self.HEADERS,
                params={"q": self.QUERY, "per_page": self.PER_PAGE, "page": page}
            )
            
            rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
            rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", time.time()))
            
            if response.status_code == 403 and rate_limit_remaining == 0:
                reset_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rate_limit_reset))
                seconds_to_reset = rate_limit_reset - time.time() + 1
                if seconds_to_reset > 0:
                    self.logger.warning(f"Rate limit reached. Reset at {reset_time}. Waiting... {seconds_to_reset:.3f} seconds.")  
                else:
                    seconds_to_reset = 10
                    self.logger.warning(f"Clock misalligmnent on rate reset time, waiting {seconds_to_reset} seconds.")
                time.sleep(seconds_to_reset) 
                continue

            if response.status_code != 200:
                self.logger.error(f"Error: {response.status_code}, {response.json()}")
                break

            data = response.json()
            items = data.get("items", [])
            results.extend(items)

            if not("incomplete_results" in data and not data["incomplete_results"]):
                break

            page += 1

            if len(items) < self.PER_PAGE:
                break
        
        return results[:self.MAX_RESULTS]

    def save_results(self, results):
        """
        Save the search results to a JSON file.

        Args:
            results (list): The results to save.
            file_path (str): The file path where the results will be saved.
        """
        
        with open(self.results_file_path, 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Saved {len(results)} results to {self.results_file_path}")

# Usage example
if __name__ == "__main__":
    github_token = os.getenv('GITHUB_TOKEN')
    notebook_search = GitHubNotebookSearch(
        token=github_token, 
        query="extension:ipynb", 
        per_page=100, 
        max_results=1000 #200000
    )
    search_results = notebook_search.search_notebooks()
    notebook_search.save_results(search_results)
