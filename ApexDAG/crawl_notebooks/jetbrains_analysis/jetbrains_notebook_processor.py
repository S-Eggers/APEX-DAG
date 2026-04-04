import json
import os
import requests
import time
import nbformat
from tqdm import tqdm

from ApexDAG.crawl_notebooks.notebook_processor import NotebookProcessor


class JetbrainsNotebookProcessor(NotebookProcessor):
    # Class-level caches for regex patterns
    alias_pattern_cache = {}
    object_pattern_cache = {}
    attribute_pattern_cache = {}

    def __init__(self, json_file, bucket_url, save_dir, log_file):
        super().__init__(save_dir, log_file)
        self.json_file = json_file
        self.bucket_url = bucket_url

    def get_notebook_code(self, url):
        notebook = self.get_notebook(url)
        return self.extract_code(notebook)

    def get_notebook(self, url):
        """
        Fetch notebook code from a URL.
        """
        try:
            time.sleep(0.01)
            response = requests.get(url, stream=True)
            response.raise_for_status()

            notebook_content = response.content.decode("utf-8")
            notebook = nbformat.reads(notebook_content, as_version=4)
            return notebook
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Failed to fetch {url}: {e}")
            return None
        except nbformat.reader.NotJSONError:
            self.logger.warning(
                f"Fetched {url} but failed to parse: Content is not a valid JSON notebook. Possible HTML content received."
            )
            return None
        except UnicodeDecodeError:
            self.logger.warning(
                f"Fetched {url} but failed to parse: 'utf-8' codec can't decode"
            )
            return None
        except Exception as e:
            self.logger.warning(f"Error processing notebook {url}: {e}", exc_info=True)
            return None

    def download_and_mine_notebooks(
        self,
        output_file_name="annotated_test.json",
        start_limit=0,
        end_limit=None,
        delay=0,
    ):
        """
        Download and process Jupyter notebooks.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        filenames = self.load_filenames(self.json_file)
        if not filenames:
            self.logger.warning("No filenames found.")
            return

        filenames = filenames[start_limit:end_limit]

        num_files = len(filenames)
        log_points = [int(num_files * i / 10) for i in range(1, 10)]

        all_annotations = {}
        for file_index, filename in tqdm(
            enumerate(filenames), total=len(filenames), desc="Processing notebooks"
        ):
            try:
                if file_index in log_points:
                    self.logger.info(
                        f" Processed {file_index + 1}/{len(filenames)} notebooks."
                    )

                file_url = f"{self.bucket_url}{filename}"
                code = self.get_notebook_code(file_url)
                if code is None:
                    continue
                annotations_object = self.process_cells(code)
                all_annotations[filename] = annotations_object
                if delay:
                    time.sleep(delay)
            except Exception as e:
                self.logger.warning(f"Error processing {filename}: {e}")
                continue

        output_file_path = os.path.join(self.save_dir, output_file_name)
        try:
            with open(output_file_path, "w") as f:
                json.dump(all_annotations, f, indent=4)
            self.logger.info(f"Annotations successfully saved to {output_file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to save annotations: {e}")

    def iterate_over_notebooks(self, start_limit=0, end_limit=None, delay=0):
        """
        Download and process Jupyter notebooks.
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        filenames = self.load_filenames(self.json_file)
        if not filenames:
            self.logger.warning("No filenames found.")
            return

        filenames = filenames[start_limit:end_limit]

        num_files = len(filenames)
        log_points = [int(num_files * i / 10) for i in range(1, 10)]

        for file_index, filename in tqdm(
            enumerate(filenames), total=len(filenames), desc="Processing notebooks"
        ):
            try:
                if file_index in log_points:
                    self.logger.info(
                        f"Processed {file_index + 1}/{len(filenames)} notebooks."
                    )

                file_url = f"{self.bucket_url}{filename}"
                notebook = self.get_notebook(file_url)

                if notebook is None:
                    continue

                yield filename

                if delay:
                    time.sleep(delay)

            except Exception as e:
                self.logger.warning(f"Error processing {filename}: {e}")
                continue


if __name__ == "__main__":
    JSON_FILE = "data/ntbs_list.json"
    BUCKET_URL = "https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/"
    SAVE_DIR = "data/notebooks"
    START_LIMIT = 50
    END_LIMIT = 80

    processor = JetbrainsNotebookProcessor(
        JSON_FILE,
        BUCKET_URL,
        SAVE_DIR,
        log_file=f"notebook_processor_{START_LIMIT}_{END_LIMIT}.log",
    )
    processor.download_and_mine_notebooks(start_limit=START_LIMIT, end_limit=END_LIMIT)
