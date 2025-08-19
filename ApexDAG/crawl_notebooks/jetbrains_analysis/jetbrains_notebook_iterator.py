import os
import nbformat
from tqdm import tqdm
from ApexDAG.crawl_notebooks.utils import InvalidNotebookException
from ApexDAG.crawl_notebooks.jetbrains_analysis.jetbrains_notebook_processor import (
    JetbrainsNotebookProcessor,
)


class JetbrainsNotebookIterator(JetbrainsNotebookProcessor):
    def __init__(
        self, JSON_FILE, BUCKET_URL, SAVE_DIR, log_file, start_index=0, stop_index=None
    ):
        super().__init__(JSON_FILE, BUCKET_URL, SAVE_DIR, log_file)
        self.filenames = self.load_filenames(self.json_file)

        self.stop_index = stop_index if stop_index else len(self.filenames)

        self.filenames = self.filenames[start_index:stop_index]
        self.progress_bar = tqdm(total=len(self.filenames), desc="Processing Notebooks")
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.filenames):
            self.progress_bar.close()
            raise StopIteration

        filename = self.filenames[self.current_index]
        self.current_index += 1
        self.progress_bar.update(1)
        notebook = self._fetch_notebook(filename)

        if notebook is not None:
            return filename, notebook
        return self.__next__()

    def _fetch_notebook(self, filename):
        """Process a single notebook, with a retry mechanism."""
        local_notebook_path = os.path.join(self.save_dir, "downloaded_notebooks", filename)

        # Check if the notebook is already cached locally
        if os.path.exists(local_notebook_path):
            self.logger.info(f"Loading notebook {filename} from cache.")
            try:
                with open(local_notebook_path, 'r', encoding='utf-8') as f:
                    return nbformat.read(f, as_version=4)
            except Exception as e:
                self.logger.warning(f"Could not read cached notebook {filename}. Refetching. Error: {e}")

        # If not cached, download it
        self.logger.info(f"Fetching notebook {filename} from remote.")
        try:
            file_url = f"{self.bucket_url}{filename}"
            notebook = self.get_notebook(file_url)
            if notebook is None:
                raise InvalidNotebookException(f"Failed to fetch notebook {filename}")
            
            # Save the downloaded notebook to the cache
            local_notebook_dir = os.path.dirname(local_notebook_path)
            if not os.path.exists(local_notebook_dir):
                os.makedirs(local_notebook_dir)

            with open(local_notebook_path, 'w', encoding='utf-8') as f:
                nbformat.write(notebook, f)

            return notebook
        except Exception as e:
            self.logger.warning(f"Attempt failed for {filename}: {e}")
            return None

    def print(self, filename="", message=""):
        message = str(message)
        self.logger.info(f"{filename}: {message}")