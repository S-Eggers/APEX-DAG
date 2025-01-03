import requests
import re
import json
import base64

from tqdm import tqdm
from ApexDAG.scripts.crawl_github.github_crawler import GitHubCrawler

class NotebookProcessor(GitHubCrawler):
    def __init__(self, notebooks):
        super().__init__()
        self.notebooks = notebooks
        self.regex_pattern_imports = r'\b(import\s+\S+|\bfrom\s+\S+\s+import\s+\S+)'
        self.regex_pattern_import_aliases = r'\b(import\s+\S+\s+as\s+\S+|\bfrom\s+\S+\s+import\s+\S+\s+as\s+\S+)'

    def download_file(self, git_url):
        """Download file content from the Git URL."""
        data =  None
        while data is None:
            response = requests.get(
                    git_url,
                    headers=self.HEADERS,
                )   
            data = self.process_response(response)
        return data
    
    def decode_base64_content(self, base64_content):
        """Decode the base64 content."""
        decoded_content = base64.b64decode(base64_content).decode('utf-8')
        return decoded_content

    def check_for_regex(self, content, regex_pattern):
        """Check the code content for regex matches."""
        matches = re.findall(regex_pattern, content)
        if matches:
            self.logger.info(f"Found matches: {matches}")
        else:
            self.logger.info("No matches found.")
            
        return matches

    def process_notebook(self, notebook):
        """Process a single notebook and check for regex matches in code cells."""
        content_data = self.download_file(notebook['git_url'])
        if content_data:
            if content_data['encoding'] == 'base64':
                decoded_content = self.decode_base64_content(content_data.get('content', ''))
            else:
                raise ValueError("Content is not Base64 encoded.")
            
        notebook['import_matches'] = self.check_for_regex(decoded_content, self.regex_pattern_imports)
        notebook['import_aliases'] = self.check_for_regex(decoded_content, self.regex_pattern_import_aliases)
        return notebook

    def process_all_notebooks(self):
        """Process all notebooks in the list."""
        for i, notebook in enumerate(tqdm(self.notebooks, desc="Processing Notebooks")):
            self.notebooks[i] = self.process_notebook(notebook)
        self.save_results(self.notebooks)
    
    def save_results(self, results):
        """Save the processed notebooks to a JSON file."""
        with open('data/raw/notebooks_imports.json', 'w') as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    with open('data/raw/notebooks.json', 'r') as f:
        notebooks = json.load(f)
    processor = NotebookProcessor(notebooks)
    processor.process_all_notebooks()