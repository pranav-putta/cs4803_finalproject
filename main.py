from dataloader import load
import pandas as pd

data = load()
df = pd.DataFrame(data['train'][:100])
print(df)