import sys
import yaml
from dataclasses import dataclass


@dataclass
class Config:
    model_name: str
    sleep_interval: int
    max_depth: int
    max_tokens: int = sys.maxsize
    llm_provider: str = "groq"
    retry_attempts: int = 3
    retry_delay: int = 30
    success_delay: int = 10


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)
