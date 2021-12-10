from datasets import load_dataset
from config import CONFIG
from transformers import AutoTokenizer


def load(dataset_name=CONFIG.dataset):
    return load_dataset(dataset_name[0])


def process_input(inp):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    out = tokenizer(inp, padding='do_not_pad', return_tensors='pt')
    return out['input_ids'], out['attention_mask']
