from datasets import load_dataset
from config import CONFIG


def load(dataset_name=CONFIG.dataset):
    return load_dataset(dataset_name[0])
