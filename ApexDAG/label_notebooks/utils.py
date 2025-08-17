import yaml
from dataclasses import dataclass


@dataclass
class Config:
    model_name: str
    sleep_interval: int
    max_depth: int


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)
