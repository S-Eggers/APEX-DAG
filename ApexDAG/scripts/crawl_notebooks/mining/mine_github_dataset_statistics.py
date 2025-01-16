
from tqdm import tqdm

from ApexDAG.scripts.crawl_notebooks.github_crawler.github_repository_notebook_iterator import GithubRepositoryNotebookIterator
from ApexDAG.scripts.crawl_notebooks.notebook_miner import NotebookMiner

if __name__ == "__main__":
    SAVE_DIR = "data/notebooks/github"
    GITHUB_API_URL = "https://api.github.com/search/code"
    START_INDEX = 0
    STOP_INDEX = 100

    miner = NotebookMiner(
        iterator = GithubRepositoryNotebookIterator(
        query="extension:ipynb",
        per_page=100,
        max_results=STOP_INDEX - START_INDEX + 1,
        search_url=GITHUB_API_URL,
        log_file=f'notebook_stat_miner_{START_INDEX}_{STOP_INDEX}.log'
    ),
        save_dir=SAVE_DIR,
        log_file=f'notebook_stat_miner_{START_INDEX}_{STOP_INDEX}.log',
        start_index=START_INDEX,
        stop_index=STOP_INDEX
    )

    miner.download_and_mine_notebooks(output_file_name="annotated_notebooks_github.json")
