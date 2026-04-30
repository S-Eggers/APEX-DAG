import json
import logging
import time

import nbformat
import requests

from ApexDAG.util.logger import configure_apexdag_logger

configure_apexdag_logger()
logger = logging.getLogger(__name__)


class JetbrainsNotebookIterator:
    """Stateful iterator that streams notebooks from the JetBrains S3 bucket."""

    def __init__(self, json_file: str, bucket_url: str, start_index: int = 0) -> None:
        self.bucket_url = bucket_url
        self.current_index = start_index

        logger.info(f"Loading filename registry from {json_file}...")
        try:
            with open(json_file, encoding="utf-8") as f:
                self.filenames = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load notebook registry: {e}")
            self.filenames = []

    def __iter__(self) -> "JetbrainsNotebookIterator":
        return self

    def __next__(self) -> tuple[str, dict]:
        while self.current_index < len(self.filenames):
            filename = self.filenames[self.current_index]
            self.current_index += 1

            notebook = self._fetch_notebook(filename)
            if notebook is not None:
                return filename, notebook

        raise StopIteration

    def _fetch_notebook(self, filename: str) -> dict | None:
        url = f"{self.bucket_url}{filename}"
        try:
            # Gentle delay to prevent Amazon S3 rate-limiting
            time.sleep(0.05)
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()

            notebook_content = response.content.decode("utf-8")
            return nbformat.reads(notebook_content, as_version=4)

        except requests.exceptions.RequestException as e:
            logger.debug(f"Network error fetching {filename}: {e}")
        except nbformat.reader.NotJSONError:
            logger.debug(f"File {filename} is not valid JSON.")
        except Exception as e:
            logger.debug(f"Unexpected error parsing {filename}: {e}")

        return None
