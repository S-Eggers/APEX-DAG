from tqdm import tqdm
from ApexDAG.scripts.crawl_notebooks.utils import InvalidNotebookException
from ApexDAG.scripts.crawl_notebooks.jetbrains_analysis.jetbrains_notebook_processor import JetbrainsNotebookProcessor


class JetbrainsNotebookIterator(JetbrainsNotebookProcessor):
    def __init__(self, JSON_FILE, BUCKET_URL, SAVE_DIR, log_file, start_index = 0, stop_index = None):
        super().__init__(JSON_FILE, BUCKET_URL, SAVE_DIR, log_file)
        self.filenames = self.load_filenames(self.json_file)
        
        self.current_index = start_index
        self.stop_index = stop_index if stop_index else len(self.filenames)

        self.filenames = self.filenames[start_index:stop_index]
        self.progress_bar = tqdm(total=len(self.filenames), desc="Processing Notebooks")

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= len(self.filenames):
            self.progress_bar.close()
            raise StopIteration

        filename = self.filenames[self.current_index]
        self.current_index += 1
        self.progress_bar.update(1)

        return self._fetch_notebook(filename)

    def _fetch_notebook(self, filename):
        """Process a single notebook, with a retry mechanism."""
        for attempt in range(2):  # Try up to 2 times
            try:
                file_url = f"{self.bucket_url}{filename}"
                code = self.get_notebook_code(file_url)
                if code is None:
                    raise InvalidNotebookException(f"Failed to fetch notebook {filename}")
                annotations_object = self.process_cells(code)
                return filename, annotations_object
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {filename}: {e}")
        self.logger.warning(f"All attempts failed for {filename}")
        return filename, None
    
    def print(self, filename, message):
        self.logger.info(f"{filename}: {message}")
        