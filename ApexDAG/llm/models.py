import sys
from dataclasses import dataclass

import yaml


@dataclass
class Config:
    llm_provider: str = "google"
    model_name: str = "gemini-3.1-flash-lite-preview"
    sleep_interval: int = 0
    max_depth: int = 5
    max_tokens: int = sys.maxsize
    retry_attempts: int = 3
    retry_delay: int = 30
    success_delay: int = 10
    max_workers: int = 2
    max_rpm: int = 12
    batch_size: int = 5


def load_config(config_path: str) -> Config:
    with open(config_path) as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)
