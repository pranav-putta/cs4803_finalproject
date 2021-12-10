from dataloader import load, process_input
import pandas as pd
from config import CONFIG
from network import ClusterFormer

data = load()
questions = data['train']['question']

input_ids, attention_mask = process_input(questions[0])
net = ClusterFormer()

print(input_ids)
output = net(input_ids.T)

print(output)
print(input_ids.shape)
print(output.shape)
