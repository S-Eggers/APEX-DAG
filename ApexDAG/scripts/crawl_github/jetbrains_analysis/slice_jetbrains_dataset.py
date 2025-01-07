import concurrent.futures
from ApexDAG.scripts.crawl_github.jetbrains_analysis.notebook_processor import NotebookProcessor
def process_slice(start_limit, end_limit):
    processor = NotebookProcessor(JSON_FILE, BUCKET_URL, SAVE_DIR)
    processor.download_notebooks(start_limit=start_limit, end_limit=end_limit)

if __name__ == "__main__":
    JSON_FILE = "data/ntbs_list.json"
    BUCKET_URL = "https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/"
    SAVE_DIR = "data/notebooks"

    # Define the slices
    slices = [
        (0, 50),
        (50, 100),
        (100, 150),
        (150, 200)
    ]

    # Use ThreadPoolExecutor to run the slices in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_slice, start, end) for start, end in slices]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")