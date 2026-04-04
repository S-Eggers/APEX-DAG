import requests
import time
import nbformat
import os
import base64
import logging
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class GitHubCrawler:
    def __init__(self, logging_file_path="app.log"):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(logging_file_path, mode="a")],
        )
        token = os.getenv("GITHUB_TOKEN")

        self.logger = logging.getLogger(__name__)
        self.GITHUB_TOKEN = token
        self.headers = {
            "Authorization": f"token {self.GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }

    def process_response(self, response):
        """
        To be used in the while loop during crawling for subclasses.
        Returns None if the rate limit is reached.
        """
        rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
        rate_limit_reset = int(response.headers.get("X-RateLimit-Reset", time.time()))

        server_time_str = response.headers.get("Date")
        if server_time_str:
            server_time = datetime.strptime(server_time_str, "%a, %d %b %Y %H:%M:%S %Z")
            server_time_unix = int(server_time.timestamp())
        else:
            server_time_unix = int(time.time())

        if response.status_code == 403 and rate_limit_remaining == 0:
            seconds_to_reset = rate_limit_reset - server_time_unix + 1

            if seconds_to_reset > 0:
                reset_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(rate_limit_reset)
                )
                self.logger.warning(
                    f"Rate limit reached. Reset at {reset_time}. Waiting... {seconds_to_reset:.3f} seconds."
                )
            else:
                seconds_to_reset = 10
                self.logger.warning(
                    f"Clock misalignment on rate reset time, waiting {seconds_to_reset} seconds."
                )

            time.sleep(seconds_to_reset)
            return None

        return response

    def download_file(self, git_url):
        """Download file content from the Git URL."""
        data = None
        while data is None:
            response = requests.get(
                git_url,
                headers=self.headers,
            )
            data = self.process_response(response)
        return data

    def decode_base64_content(self, base64_content):
        """Decode the base64 content."""
        decoded_content = base64.b64decode(base64_content).decode("utf-8")
        return decoded_content

    def decode(self, response_data):
        if response_data:
            if response_data["encoding"] == "base64":
                decoded_content = self.decode_base64_content(
                    response_data.get("content", "")
                )
            else:
                raise ValueError(
                    f"Content is not Base64 encoded, but {response_data.get('encoding', '')}."
                )
            return decoded_content
        raise ValueError("Content does not exist.")

    def get_notebook(self, url):
        """
        Fetch notebook code from a URL.
        """
        try:
            response = self.download_file(url)
            notebook_content = self.decode(response.json())
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
        except Exception as e:
            self.logger.warning(f"Error processing notebook {url}: {e}", exc_info=True)
            return None
