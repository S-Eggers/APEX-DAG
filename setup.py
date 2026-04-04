import json
import os
from setuptools import setup, find_packages

package_json_path = os.path.join(
    os.path.dirname(__file__), "apex-dag-jupyter", "package.json"
)

with open(package_json_path, "r") as f:
    package_data = json.load(f)
    version = package_data.get("version", "0.0.1")

setup(
    name="ApexDAG",
    version=version,
    packages=find_packages(include=["ApexDAG", "ApexDAG.*"]),
    install_requires=[
        "nbformat==5.10.4",
        "networkx==3.3",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "tqdm==4.66.4",
        "ipykernel==6.29.5",
        "python-dotenv==1.0.1",
        "fasttext==0.9.3",
        "pydantic==2.10.6",
        "dotenv==0.9.9",
        "wandb",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            # Define command-line scripts here
        ],
    },
)
