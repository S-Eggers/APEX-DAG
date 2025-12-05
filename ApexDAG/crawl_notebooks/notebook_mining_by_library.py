"""
ML Notebook Scraper
Downloads Jupyter notebooks from GitHub organized by ML library.
50 notebooks per library, with size limits and rate limiting.
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

GITHUB_TOKEN: Optional[str] = os.getenv("GITHUB_TOKEN")
# Download settings
NOTEBOOKS_PER_LIBRARY = 50
MAX_FILE_SIZE_KB = 50  # Skip notebooks larger than this
MIN_FILE_SIZE_KB = 2    # Skip notebooks smaller than this (avoid empty/trivial notebooks)
OUTPUT_DIR = "/home/nina/projects/APEX-DAG/data/notebook_data"

# Rate limiting (GitHub allows 30 search requests/min authenticated, 10 unauth)
SEARCH_DELAY_SECONDS = 2.5
DOWNLOAD_DELAY_SECONDS = 0.5

LIBRARY_SEARCHES = {
    # DL libraries
    "pytorch": [
        '"import torch"',
        '"from torch"',
    ],
    "tensorflow": [
        '"import tensorflow"',
        '"from tensorflow"',
    ],
    "keras": [
        '"import keras"',
        '"from keras"',
        '"from tensorflow.keras"',
    ],
    "jax": [
        '"import jax"',
        '"from jax"',
        '"import flax"',
    ],
    "mxnet": [
        '"import mxnet"',
        '"from mxnet"',
    ],
    "paddlepaddle": [
        '"import paddle"',
        '"from paddle"',
    ],
    
    # STats/ classical ML algorithms
    "sklearn": [
        '"import sklearn"',
        '"from sklearn"',
    ],
    "xgboost": [
        '"import xgboost"',
        '"from xgboost"',
    ],
    "lightgbm": [
        '"import lightgbm"',
        '"from lightgbm"',
    ],
    "catboost": [
        '"import catboost"',
        '"from catboost"',
    ],
    "statsmodels": [
        '"import statsmodels"',
        '"from statsmodels"',
    ],
    
    # NLP
    "transformers": [
        '"from transformers"',
        '"import transformers"',
    ],
    "spacy": [
        '"import spacy"',
        '"from spacy"',
    ],
    "nltk": [
        '"import nltk"',
        '"from nltk"',
    ],
    "gensim": [
        '"import gensim"',
        '"from gensim"',
    ],
    "flair": [
        '"import flair"',
        '"from flair"',
    ],
    
    # CV
    "opencv": [
        '"import cv2"',
        '"from cv2"',
    ],
    "torchvision": [
        '"import torchvision"',
        '"from torchvision"',
    ],
    "timm": [
        '"import timm"',
        '"from timm"',
    ],
    "detectron2": [
        '"import detectron2"',
        '"from detectron2"',
    ],
    "albumentations": [
        '"import albumentations"',
        '"from albumentations"',
    ],
    "ultralytics": [
        '"from ultralytics"',
        '"import ultralytics"',
    ],
    
    # AutoML
    "autogluon": [
        '"import autogluon"',
        '"from autogluon"',
    ],
    "h2o": [
        '"import h2o"',
        '"from h2o"',
    ],
    "pycaret": [
        '"import pycaret"',
        '"from pycaret"',
    ],
    "optuna": [
        '"import optuna"',
        '"from optuna"',
    ],
    "ray_tune": [
        '"from ray.tune"',
        '"from ray import tune"',
    ],
    
    # RL
    "stable_baselines3": [
        '"import stable_baselines3"',
        '"from stable_baselines3"',
    ],
    "gymnasium": [
        '"import gymnasium"',
        '"import gym"',
    ],
    
    # GNNs
    "pytorch_geometric": [
        '"import torch_geometric"',
        '"from torch_geometric"',
    ],
    "dgl": [
        '"import dgl"',
        '"from dgl"',
    ],
    
    # Time Series
    "prophet": [
        '"from prophet"',
        '"from fbprophet"',
    ],
    "darts": [
        '"import darts"',
        '"from darts"',
    ],
    "gluonts": [
        '"import gluonts"',
        '"from gluonts"',
    ],
    "sktime": [
        '"import sktime"',
        '"from sktime"',
    ],
    
    # LLMs
    "diffusers": [
        '"import diffusers"',
        '"from diffusers"',
    ],
    "langchain": [
        '"import langchain"',
        '"from langchain"',
    ],
    "llama_index": [
        '"import llama_index"',
        '"from llama_index"',
    ],
}

@dataclass
class NotebookInfo:
    name: str
    repo: str
    path: str
    url: str


class GitHubNotebookScraper:
    """Scrapes Jupyter notebooks from GitHub by ML library."""
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: Optional[str] = None):
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "ML-Notebook-Scraper/1.0"
        })
        if token:
            self.session.headers["Authorization"] = f"token {token}"
            print("✓ Using authenticated requests (higher rate limits)")
        else:
            print("⚠ No GitHub token provided - rate limits will be strict")
            print("  Set GITHUB_TOKEN environment variable for better performance")
        
        self.downloaded_urls = set()
        self.stats = {"searched": 0, "downloaded": 0, "skipped_size": 0, "skipped_small": 0, "skipped_dup": 0, "errors": 0}
        self.max_file_size_kb = MAX_FILE_SIZE_KB  # Can be overridden
        self.min_file_size_kb = MIN_FILE_SIZE_KB  # Can be overridden
    
    def search_notebooks(self, query: str, per_page: int = 30) -> list[NotebookInfo]:
        """Search GitHub for notebooks matching query."""
        full_query = f'{query} language:"Jupyter Notebook"'
        
        params = {
            "q": full_query,
            "per_page": min(per_page, 100),
            "sort": "indexed",
        }
        
        try:
            response = self.session.get(
                f"{self.BASE_URL}/search/code",
                params=params
            )
            
            # handle rate limiting
            if response.status_code == 403:
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                wait_time = max(reset_time - time.time(), 60)
                print(f"  ⏳ Rate limited. Waiting {wait_time:.0f}s...")
                time.sleep(wait_time + 1)
                return self.search_notebooks(query, per_page)
            
            response.raise_for_status()
            data = response.json()
            
            notebooks = []
            for item in data.get("items", []):
                repo = item["repository"]["full_name"]
                path = item["path"]
                
                notebooks.append(NotebookInfo(
                    name=item["name"],
                    repo=repo,
                    path=path,
                    url=item["html_url"],
                ))
            
            return notebooks
            
        except requests.exceptions.RequestException as e:
            print(f"  ✗ Search error: {e}")
            self.stats["errors"] += 1
            return []
    
    def download_notebook(self, notebook: NotebookInfo, output_path: Path) -> bool:
        """Download a single notebook using GitHub Contents API."""
        try:
            contents_url = f"{self.BASE_URL}/repos/{notebook.repo}/contents/{notebook.path}"
            response = self.session.get(contents_url)
            
            if response.status_code == 404:
                self.stats["errors"] += 1
                return False
            
            # handle rate limiting
            if response.status_code == 403:
                reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                wait_time = max(reset_time - time.time(), 60)
                print(f"  ⏳ Rate limited. Waiting {wait_time:.0f}s...")
                time.sleep(wait_time + 1)
                return self.download_notebook(notebook, output_path)
            
            if response.status_code != 200:
                print(f"     ✗ Failed to get file info (status {response.status_code}): {notebook.name}")
                self.stats["errors"] += 1
                return False
            
            file_info = response.json()
            
            file_size = file_info.get("size", 0)
            if file_size > self.max_file_size_kb * 1024:
                self.stats["skipped_size"] += 1
                print(f"     ✗ Skipped large file: {notebook.name} ({file_size} bytes)")
                return False
            
            if file_size < self.min_file_size_kb * 1024:
                self.stats["skipped_small"] += 1
                print(f"     ✗ Skipped small file: {notebook.name} ({file_size} bytes)")
                return False
            
            # real download URL from Contents API
            download_url = file_info.get("download_url")
            if not download_url:
                self.stats["errors"] += 1
                return False
            
            # actual content
            time.sleep(DOWNLOAD_DELAY_SECONDS)
            content_response = self.session.get(download_url)
            
            if content_response.status_code != 200:
                print(f"     ✗ Failed to download (status {content_response.status_code}): {notebook.name}")
                self.stats["errors"] += 1
                return False
            
            content = content_response.content
            
            try:
                json.loads(content)
            except json.JSONDecodeError:
                self.stats["errors"] += 1
                return False
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(content)
            
            self.stats["downloaded"] += 1
            return True
            
        except Exception as e:
            print(f"     ✗ Error downloading {notebook.name}: {e}")
            self.stats["errors"] += 1
            return False
    
    def scrape_library(self, library_name: str, search_terms: list[str], 
                       output_dir: Path, target_count: int) -> int:
        """Scrape notebooks for a single library."""
        library_dir = output_dir / library_name
        library_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = 0
        seen_repos = {}
        
        print(f"\n{'='*60}")
        print(f"📚 {library_name.upper()}")
        print(f"{'='*60}")
        
        for search_term in search_terms:
            if downloaded >= target_count:
                break
            
            print(f"\n Searching: {search_term}")
            time.sleep(SEARCH_DELAY_SECONDS)
            self.stats["searched"] += 1
            
            notebooks = self.search_notebooks(search_term, per_page=100)
            print(f"     Found {len(notebooks)} results")
            
            for nb in notebooks:
                if downloaded >= target_count:
                    break
                
                unique_key = f"{nb.repo}/{nb.path}"
                if unique_key in self.downloaded_urls:
                    self.stats["skipped_dup"] += 1
                    continue
                
                if seen_repos.get(nb.repo, 0) >= 2:
                    continue
                
                safe_name = f"{nb.repo.replace('/', '_')}_{nb.name}"
                safe_name = "".join(c for c in safe_name if c.isalnum() or c in "._-")[:100]
                if not safe_name.endswith(".ipynb"):
                    safe_name += ".ipynb"
                
                output_path = library_dir / safe_name
                
                if output_path.exists():
                    downloaded += 1
                    self.downloaded_urls.add(unique_key)
                    continue
                
                time.sleep(DOWNLOAD_DELAY_SECONDS)
                
                if self.download_notebook(nb, output_path):
                    downloaded += 1
                    self.downloaded_urls.add(unique_key)
                    seen_repos[nb.repo] = seen_repos.get(nb.repo, 0) + 1
                    print(f"     ✓ [{downloaded}/{target_count}] {nb.name[:50]}")
        
        print(f"\n  Downloaded {downloaded} notebooks to {library_dir}")
        return downloaded
    
    def scrape_all(self, output_dir: str = OUTPUT_DIR, 
                   target_per_library: int = NOTEBOOKS_PER_LIBRARY,
                   libraries: Optional[list[str]] = None):
        """Scrape notebooks for all configured libraries."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        libs_to_scrape = libraries or list(LIBRARY_SEARCHES.keys())
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║           ML NOTEBOOK SCRAPER - GitHub Edition               ║
╠══════════════════════════════════════════════════════════════╣
║  Target: {target_per_library} notebooks per library                          ║
║  Libraries: {len(libs_to_scrape):3d}                                             ║
║  File size range: {MIN_FILE_SIZE_KB}-{MAX_FILE_SIZE_KB} KB                               ║
║  Output: {str(output_path):<50} ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        start_time = datetime.now()
        results = {}
        
        for library_name in libs_to_scrape:
            if library_name not in LIBRARY_SEARCHES:
                print(f"⚠ Unknown library: {library_name}")
                continue
            
            search_terms = LIBRARY_SEARCHES[library_name]
            count = self.scrape_library(
                library_name, search_terms, output_path, target_per_library
            )
            results[library_name] = count
        
        # summary
        elapsed = datetime.now() - start_time
        total_downloaded = sum(results.values())
        
        print(f"""
╔══════════════════════════════════════════════════════════════╗
║                        SUMMARY                               ║
╠══════════════════════════════════════════════════════════════╣
║  Total notebooks downloaded: {total_downloaded:5d}                           ║
║  Search queries made:        {self.stats['searched']:5d}                           ║
║  Skipped (too large):        {self.stats['skipped_size']:5d}                           ║
║  Skipped (too small):        {self.stats['skipped_small']:5d}                           ║
║  Skipped (duplicates):       {self.stats['skipped_dup']:5d}                           ║
║  Errors:                     {self.stats['errors']:5d}                           ║
║  Time elapsed:               {str(elapsed).split('.')[0]:>10}                    ║
╚══════════════════════════════════════════════════════════════╝
        """)
        
        print("\n Per-Library Results:")
        print("-" * 40)
        for lib, count in sorted(results.items(), key=lambda x: -x[1]):
            bar = "█" * (count // 2) + "░" * ((target_per_library - count) // 2)
            print(f"  {lib:20s} {count:3d}/{target_per_library} {bar}")
        
        # Save metadata
        metadata = {
            "scraped_at": datetime.now().isoformat(),
            "target_per_library": target_per_library,
            "max_file_size_kb": MAX_FILE_SIZE_KB,
            "results": results,
            "stats": self.stats
        }
        metadata_path = output_path / "scrape_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"\n📄 Metadata saved to {metadata_path}")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Jupyter notebooks from GitHub by ML library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all libraries (50 notebooks each)
  python ml_notebook_scraper.py

  # Download specific libraries
  python ml_notebook_scraper.py --libraries pytorch sklearn transformers

  # Custom settings
  python ml_notebook_scraper.py --count 100 --max-size 1000 --output ./data

  # List available libraries
  python ml_notebook_scraper.py --list
        """
    )
    
    parser.add_argument(
        "--output", "-o", 
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--count", "-n", 
        type=int, 
        default=NOTEBOOKS_PER_LIBRARY,
        help=f"Notebooks per library (default: {NOTEBOOKS_PER_LIBRARY})"
    )
    parser.add_argument(
        "--max-size", "-s", 
        type=int, 
        default=MAX_FILE_SIZE_KB,
        help=f"Max file size in KB (default: {MAX_FILE_SIZE_KB})"
    )
    parser.add_argument(
        "--libraries", "-l", 
        nargs="+",
        help="Specific libraries to scrape (default: all)"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List available libraries and exit"
    )
    parser.add_argument(
        "--token", "-t",
        help="GitHub token (or set GITHUB_TOKEN env var)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable libraries:")
        print("-" * 40)
        for lib in sorted(LIBRARY_SEARCHES.keys()):
            terms = ", ".join(LIBRARY_SEARCHES[lib][:2])
            print(f"  {lib:20s} ({terms}...)")
        return
    
    token = args.token or GITHUB_TOKEN
    scraper = GitHubNotebookScraper(token=token)
    scraper.max_file_size_kb = args.max_size
    scraper.scrape_all(
        output_dir=args.output,
        target_per_library=args.count,
        libraries=args.libraries
    )


if __name__ == "__main__":
    main()