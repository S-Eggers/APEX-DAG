import requests
import time
import os
import json
import logging
from dotenv import load_dotenv
from ApexDAG.scripts.crawl_github.github_crawler import GitHubCrawler

load_dotenv()

class GitHubNotebookSearch(GitHubCrawler):
    def __init__(self, query: str, per_page: int = 30, max_results: int = 10, results_file_path: str = 'data/raw/notebooks.json'):
        super().__init__()
        self.QUERY = query
        self.PER_PAGE = per_page
        self.MAX_RESULTS = max_results
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
            
            data = self.process_response(response)
            
            if data is not None:
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
    notebook_search = GitHubNotebookSearch(
        query="extension:ipynb", 
        per_page=100, 
        max_results=1000 #200000
    )
    search_results = notebook_search.search_notebooks()
    notebook_search.save_results(search_results)
