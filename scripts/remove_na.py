import pandas as pd

file_path = 'C:\Projects\CreativitySmartScoring\outputs\pku_out.csv'
df = pd.read_csv(file_path)
df = df.dropna(how='any')
df.to_csv(file_path, index=False)