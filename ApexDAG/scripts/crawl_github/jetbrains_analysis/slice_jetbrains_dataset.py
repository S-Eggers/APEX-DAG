import concurrent.futures
from ApexDAG.scripts.crawl_github.jetbrains_analysis.notebook_processor import NotebookProcessor


def create_slices(start, end, step):
    """
    Create slices from start to end with the given step.
    """
    return [(i, min(i + step, end)) for i in range(start, end, step)]

def process_slice(start_limit, end_limit):
    processor = NotebookProcessor(JSON_FILE, BUCKET_URL, SAVE_DIR, log_file=f"notebooks_{start_limit}_{end_limit}.log")
    processor.download_notebooks(start_limit=start_limit, end_limit=end_limit, output_file_name=f"notebooks_{start_limit}_{end_limit}.json")

if __name__ == "__main__":
    JSON_FILE = "data/ntbs_list.json" # part of jetbrains dataset
    BUCKET_URL = "https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/"
    SAVE_DIR = "data/notebooks/paralell/"

    # Define the slices for toy example
    slices = create_slices(0, 10000, 500)
    print(f"Creating {len(slices)} slices.")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_slice, start, end) for start, end in slices]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")