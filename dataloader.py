from datasets import load_dataset
from config import CONFIG
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


def load(dataset_name=CONFIG.dataset):
    return load_dataset(dataset_name[0])


def process_input(inp, segment=0):
    out = tokenizer(inp)
    return out['input_ids'], segment * np.ones(len(out['token_type_ids']))
