import argparse
import os
import concurrent.futures
from ApexDAG.crawl_notebooks.jetbrains_analysis.jetbrains_notebook_processor import JetbrainsNotebookProcessor

def create_slices(start, end, step):
    """
    Create slices from start to end with the given step.
    """
    return [(i, min(i + step, end)) for i in range(start, end, step)]

def process_slice(start_limit, end_limit, json_file, bucket_url, save_dir):
    processor = JetbrainsNotebookProcessor(
        json_file, bucket_url, save_dir, log_file=f"notebooks_{start_limit}_{end_limit}.log"
    )
    processor.download_and_mine_notebooks(
        start_limit=start_limit, 
        end_limit=end_limit, 
        output_file_name=f"notebooks_{start_limit}_{end_limit}.json"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process notebook slices in parallel.")
    parser.add_argument("--json_file", type=str, default="data/ntbs_list.json", help="Path to the JSON file.")
    parser.add_argument("--bucket_url", type=str, default="https://github-notebooks-update1.s3-eu-west-1.amazonaws.com/", help="Bucket URL for downloading notebooks.")
    parser.add_argument("--save_dir", type=str, default="/home/nina/data/apexdag_results/jetbrains_stats_10m/full", help="Directory to save the notebooks.")
    parser.add_argument("--start", type=int, default=0, help="Start index for processing.")
    parser.add_argument("--end", type=int, default=10000000, help="End index for processing.")
    parser.add_argument("--step", type=int, default=50000, help="Step size for slices.")

    args = parser.parse_args()

    JSON_FILE = args.json_file
    BUCKET_URL = args.bucket_url
    SAVE_DIR = args.save_dir
    START = args.start
    END = args.end
    STEP = args.step

    slices = create_slices(START, END, STEP)
    
    print(f"Creating {len(slices)} slices.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() * 5) as executor:
        futures = [
            executor.submit(process_slice, start, end, JSON_FILE, BUCKET_URL, SAVE_DIR)
            for start, end in slices
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error occurred: {e}")

        executor.shutdown(wait=True)
