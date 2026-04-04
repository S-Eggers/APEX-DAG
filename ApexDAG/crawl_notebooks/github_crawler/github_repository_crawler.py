"""
Github's API prohibits fetching more than 1000 query outputs - code or repositories.

In this project, we are interested in code. However, there is a twist: repositories can be filtered by push
dates. They can also be sorted by update date. By using different filtering clauses on repositories and sorting them in a clever way,
we can get way more than 1000 results.

The approach involves:
1. Filtering repositories by push date
2. Sorting repositories by update date. # since it is very similar to push
3. Creating subqueries for different date ranges to bypass the 1000 results limit. # the subqueries in our configuration should quasi-non-overlaping
4. Using pagination to fetch all pages of results for each subquery.

This method allows us to systematically retrieve a large number of repositories beyond the 1000 results limit imposed by the API.
"""

import json
import time
import requests
import os
import argparse
import pandas as pd
import logging
from tqdm import tqdm
from datetime import datetime
from requests.exceptions import HTTPError

DELAY_BETWEEN_QUERIES = 2


class GitHubRepositoryCrawler:
    def __init__(
        self,
        query,
        last_acceptable_date,
        log_file="log_repo.txt",
        filter_date_start="2024-10-01",
        filter_date_end="2025-01-15",
        save_folder="",
    ):
        self.token = os.getenv("GITHUB_TOKEN")
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
        }

        self.url = "https://api.github.com/search/repositories?q="
        self.filter = "pushed"  # pushed works the best, since we can then also then filter by updated
        self.sort = "updated"  # field by which we sort the retrieved items
        self.query = query

        self.max_depth = 5
        self.timeout = 15

        self.items_per_page = 100
        self.accepted_languages = ["Jupyter Notebook"]

        self.hash_dict = (
            dict()
        )  # storing (user, repository) and the value will be filename

        self.filter_date_start = filter_date_start
        self.filter_date_end = filter_date_end

        self.subqueries = self.create_subqueries(
            self.filter_date_start, self.filter_date_end
        )
        self.visited = set()

        self.last_acceptable_date = last_acceptable_date

        self.number_of_repos = 0

        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)
        name_of_query = (
            self.query.replace(" ", "_") if self.query.replace(" ", "_") else "all"
        )

        self.result_file = f"result_{name_of_query}_{self.filter_date_start}_{self.filter_date_end}.json"
        self.result_file = os.path.join(save_folder, self.result_file)

    def create_subqueries(self, start_date, end_date):
        """
        Create subqueries for the given date range, witha a monthly frequency.

        Args:
            start_date (str): The start date in 'YYYY-MM-DD' format.
            end_date (str): The end date in 'YYYY-MM-DD' format.

        Returns:
            list: A list of subquery strings.
        """
        subqueries = []
        dates = (
            pd.date_range(start_date, end_date, freq="ME").strftime("%Y-%m-%d").tolist()
        )

        prefix = (
            "+" if self.query else ""
        )  # we only need the + if there is something that we are searching fo - a subset
        for date in dates:
            for lang in self.accepted_languages:
                subqueries.append(
                    f"{prefix}language:{lang}+{self.filter}%3A<%3D{date}&per_page={self.items_per_page}&sort={self.sort}&order=desc"
                )
                subqueries.append(
                    f"{prefix}language:{lang}+{self.filter}%3A>%3D{date}&per_page={self.items_per_page}&sort={self.sort}&order=asc"
                )
        return subqueries

    def getUrl(self, url):
        """Given a URL it returns its body"""
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        return response.json()

    def createUrl(self, subquery):
        return self.url + self.query + subquery  # subqueries already have the params...

    def _get_all_pages(self, current_url, number_of_pages):
        """Given a URL it returns all the pages of the response"""
        for currentPage in tqdm(
            range(1, number_of_pages + 1), desc="Processing pages", leave=False
        ):
            url = current_url + "&page=" + str(currentPage)
            data = json.loads(json.dumps(self.getUrl(url)))
            for item in tqdm(data["items"], desc="Processing items", leave=False):
                user = item["owner"]["login"]
                repository = item["name"]
                self.number_of_repos += 1
                if (user, repository) not in self.visited:
                    self.check_for_notebook_files(user, repository)
                self.visited.add((user, repository))

            tqdm.write(f"Number of repositories read: {self.number_of_repos}")
            tqdm.write(
                f"Number of unique repositories in hashset: {len(self.hash_dict)}"
            )
            time.sleep(DELAY_BETWEEN_QUERIES)

    def crawl(self):
        """
        Crawl GitHub repositories based on the subqueries generated.

        This method iterates over the subqueries, makes requests to the GitHub API,
        retrieves repository data, and processes all pages of the response. It also
        updates the progress using a tqdm progress bar and logs the number of repositories
        read and the number of unique repositories in the hashset.

        Returns:
            None
        """
        for subquery in tqdm(self.subqueries):  # add a tdqm progress bar
            current_url = self.createUrl(subquery)
            data = json.loads(json.dumps(self.getUrl(current_url)))
            number_of_pages = max(
                1, int(min(data["total_count"], 1000) / self.items_per_page)
            )

            self._get_all_pages(current_url, number_of_pages)
            tqdm.write(f"Theoretically available in this query: {data['total_count']}")

        self.logger.info(
            "DONE! " + str(self.number_of_repos) + " repositories have been processed."
        )
        self.logger.info(
            "Real number of processed repos" + str(len(list(self.hash_dict.keys())))
        )

        # save the hashset
        with open(self.result_file, "w") as f:
            json.dump(
                {
                    f"{k[0]}/{k[1]}": {"notebook_files": v[0], "data_files": v[1]}
                    for k, v in self.hash_dict.items()
                },
                f,
            )

        return self.hash_dict

    def check_for_notebook_files(self, user, repository):
        """
        Check for Jupyter notebook and data files in the given repository, including all subdirectories.

        Args:
            user (str): GitHub username.
            repository (str): Repository name.

        Returns:
            None: Updates self.hash_dict with notebook and data file details.
        """
        try:
            api_url = f"https://api.github.com/repos/{user}/{repository}/contents"

            extensions_data = [
                ".csv",
                ".json",
                ".xls",
                ".xlsx",
                ".parquet",
                ".sql",
                ".yml",
                ".yaml",
                ".zip",
                ".tar.gz",
                ".7z",
                ".xml",
                ".h5",
                ".hdf5",
                ".pkl",
            ]
            extension_notebooks = ".ipynb"

            def fetch_all_files(url, depth=1):
                if depth == self.max_depth:
                    return []
                try:
                    response = requests.get(url, headers=self.headers)
                    response.raise_for_status()

                    files = response.json()
                    all_files = []

                    for file in files:
                        if file["type"] == "file":
                            if file["name"].endswith(extension_notebooks) or any(
                                file["name"].endswith(ext) for ext in extensions_data
                            ):
                                all_files.append(file)
                        elif file["type"] == "dir":
                            if "site-packages" in file["url"]:
                                continue  # if somebody uploaded ther environment we need to pass
                            all_files.extend(fetch_all_files(file["url"], depth + 1))

                    return all_files

                except HTTPError as http_err:
                    if response.status_code == 403:  # Rate limit exceeded
                        reset_time = int(
                            response.headers.get(
                                "X-RateLimit-Reset", time.time() + 3600
                            )
                        )
                        server_time_str = response.headers.get("Date")
                        if server_time_str:
                            server_time = datetime.strptime(
                                server_time_str, "%a, %d %b %Y %H:%M:%S %Z"
                            )
                            server_time_unix = int(server_time.timestamp())
                        else:
                            server_time_unix = int(time.time())

                        sleep_duration = reset_time - server_time_unix
                        if sleep_duration > 0:
                            print(
                                f"Rate limit exceeded. Sleeping for {sleep_duration} seconds."
                            )
                            time.sleep(sleep_duration)
                            return fetch_all_files(url, depth)  # Retry after sleeping
                    else:
                        self.logger(f"HTTP error occurred: {http_err}")
                    return None

                except Exception as e:
                    self.logger.error(f"Failed to fetch files from {url}: {e}")
                    return []

            all_files = fetch_all_files(api_url)

            data_files = {
                file["name"]: {"git_url": file["git_url"], "html_url": file["html_url"]}
                for file in all_files
                if any(file["name"].endswith(ext) for ext in extensions_data)
            }
            notebook_files = {
                file["name"]: {"git_url": file["git_url"], "html_url": file["html_url"]}
                for file in all_files
                if file["name"].endswith(extension_notebooks)
            }

            if notebook_files:
                self.hash_dict[(user, repository)] = (notebook_files, data_files)

        except Exception as e:
            self.logger.error(
                f"Failed to check for data files in repository {user}/{repository}: {e}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GitHub Repository Crawler")
    parser.add_argument(
        "--query",
        type=str,
        default="machine learning",
        help="The search query for GitHub repositories",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2020-01-31",
        help="The last acceptable date for repositories",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="data/notebooks/github",
        help="The folder to save the results",
    )
    args = parser.parse_args()

    crawler = GitHubRepositoryCrawler(args.query, args.date)
    crawler.crawl()
