from tqdm import tqdm
import requests
import json
import os
import logging
from dotenv import load_dotenv
from ApexDAG.scripts.crawl_notebooks.github_crawler.github_crawler import GitHubCrawler

load_dotenv()

class GithubRepositoryNotebookIterator(GitHubCrawler):
    def __init__(self, max_results, notebook_paths, log_file="log.txt"):
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
        
        self.notebook_paths = notebook_paths
        self.results_notebook = self._read_notebook_list()

        self.current_results = []
        self.current_index = 0

        logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        self.max_results = min(len(self.results_notebook), max_results)
        self.progress_bar = tqdm(total=self.max_results, desc="Fetching Notebooks")

    def __iter__(self):
        return self
    
    def _read_notebook_list(self):
        """
        Read the JSON object from the result file and flatten it to a list with only the notebooks.

        Returns:
            list: A list of notebook files with their details.
        """
        with open(self.notebook_paths, 'r') as f:
            data = json.load(f)
        notebook_list = []
        for _, files in data.items():
            notebook_files = files["notebook_files"]  # notebook_files is the first element in the tuple
            data_files = files["data_files"] # data_files is the second element in the
            for notebook in notebook_files.values():
                notebook['data_files'] = data_files
                notebook_list.append(notebook)
        return notebook_list

    def __next__(self):

        if self.current_index < self.max_results:
            
            notebook = self.results_notebook[self.current_index]
            self.current_index +=1
            self.progress_bar.update(1)
            
            notebook_content = self.get_notebook(notebook['git_url'])
            
            if notebook_content is None:
                return self.__next__()
            
            return notebook['html_url'],{
                "notebook_content": notebook_content,
                "data_files": notebook['data_files'],
            }
        else:
            self.progress_bar.close()
            raise StopIteration

        
    def print(self, filename = '', message = ''):
        message = str(message)
        self.logger.info(f"{filename}: {message}")
            

# Usage example
if __name__ == "__main__":
    GITHUB_API_URL = "https://api.github.com/search/code"

    
    notebook_iterator = GithubRepositoryNotebookIterator(
        
        max_results=10000,
        notebook_paths = 'result_machine_learning_2024-10-01_2025-01-15.json',
        log_file="notebook_iterator.log"
    )
