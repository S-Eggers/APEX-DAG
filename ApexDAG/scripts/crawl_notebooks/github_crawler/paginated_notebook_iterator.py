from tqdm import tqdm
import requests
import json
import os
import logging
from dotenv import load_dotenv
from ApexDAG.scripts.crawl_notebooks.github_crawler.github_crawler import GitHubCrawler

load_dotenv()

class PaginatedNotebookIterator(GitHubCrawler):
    def __init__(self, query, per_page, max_results, search_url, log_file="log.txt"):
        """
        Initialize the iterator with GitHub API details.

        Args:
            query (str): Search query for GitHub API.
            per_page (int): Number of results per page.
            max_results (int): Maximum total results to fetch.
            search_url (str): The GitHub API endpoint for code search.
            headers (dict): HTTP headers for requests.
            log_file (str): Path to the log file.
        """
        super().__init__(logging_file_path = log_file)
        
        self.query = query
        self.per_page = per_page
        self.max_results = max_results
        self.search_url = search_url

        self.current_results = []
        self.current_index = 0
        self.total_fetched = 0
        self.current_page = 1

        # set up logging
        logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        self.progress_bar = tqdm(total=max_results, desc="Fetching Notebooks")

    def __iter__(self):
        return self

    def __next__(self):
        # check if the current results batch is exhausted
        if self.current_index >= len(self.current_results):
            # fetch the next page of results
            if self.total_fetched >= self.max_results:
                self.progress_bar.close()
                raise StopIteration

            self._fetch_next_page()
            self.current_index = 0

        if self.current_results:
            notebook = self.current_results[self.current_index]
            self.current_index += 1
            self.progress_bar.update(1)
            
            notebook_content = self.get_notebook(notebook['git_url'])
            
            if notebook_content is None:
                return self.__next__()

            data_files = self.check_for_data_files(notebook['repository'])
            
            return notebook['html_url'],{
                "notebook_content": notebook_content,
                "data_files": data_files,
            }
            

        else:
            self.progress_bar.close()
            raise StopIteration

    def _fetch_next_page(self):
        """
        Fetch the next page of results from the GitHub API.
        """

        try:
            response = requests.get(
                self.search_url,
                headers=self.headers,
                params={"q": self.query, "per_page": self.per_page, "page": self.current_page}
            )
            response.raise_for_status()  # raise an exception for HTTP errors
            data = response.json()

            items = data.get("items", [])
            self.current_results = items
            self.total_fetched += len(items)

            self.logger.info(f"Fetched page {self.current_page} with {len(items)} results.")
            self.current_page += 1

            # stop if there are no more items or we've reached the max results
            if not items or self.total_fetched >= self.max_results:
                self.progress_bar.close()
                raise StopIteration
        except Exception as e:
            self.logger.error(f"Error fetching page {self.current_page}: {e}")
            self.current_results = []
            
    def check_for_data_files(self, repository):
        """
        Check for various data files in the given repository.

        Args:
            repository (dict): Repository information from the GitHub API search results.

        Returns:
            list: A list of all matching data file names.
        """
        try:
            repo_full_name = repository.get('full_name', '')
            api_url = f"https://api.github.com/repos/{repo_full_name}/contents"
            response = requests.get(api_url, headers=self.headers)
            response.raise_for_status()

            files = response.json()
            
            extensions = [
                '.csv', '.json', '.xls', '.xlsx', '.parquet', '.sql',
                '.yml', '.yaml', '.zip', '.tar.gz', '.7z', '.xml',
                '.h5', '.hdf5', '.pkl'
            ]
            
            data_files = [file['name'] for file in files if any(file['name'].endswith(ext) for ext in extensions)]
            return data_files
        except Exception as e:
            self.logger.error(f"Failed to check for data files in repository {repository.get('full_name')}: {e}")
            return []
        
    def print(self, filename = '', message = ''):
        message = str(message)
        self.logger.info(f"{filename}: {message}")
            

# Usage example
if __name__ == "__main__":
    GITHUB_API_URL = "https://api.github.com/search/code"

    
    notebook_iterator = PaginatedNotebookIterator(
        query="extension:ipynb",
        per_page=100,
        max_results=1000,
        search_url=GITHUB_API_URL,
        log_file="notebook_iterator.log"
    )
    
    for (item) in notebook_iterator:
        print(item)