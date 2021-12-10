from dataclasses import dataclass
import yaml


@dataclass
class Config:
    list_datasets: list
    select_dataset: int
    model: dict

    @property
    def dataset(self):
        return self.list_datasets[self.select_dataset]


config_file = 'config.yaml'

with open(config_file) as f:
    CONFIG = Config(**yaml.safe_load(f))
