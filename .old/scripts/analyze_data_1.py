import pandas as pd
from sentence_transformers import SentenceTransformer
from css.scorer import Scorer
from matplotlib import pyplot as plt
import envs

data_path = f'{envs.project_path}/data'
df = pd.read_csv(f'{data_path}/AUT_Bedsheet.csv')
# df.to_csv(f'{data_path}/AUT_Bedsheet.csv', encoding='utf-8-sig')

df = df.fillna('')
columns = [_ for _ in df.columns if _.startswith('Bedsheet')]

data = [df[_].tolist() for _ in columns]

all_words = ['床单'] + sorted(list(set(sum(data, []))))
model_names = ['paraphrase-multilingual-MiniLM-L12-v2',
               'paraphrase-multilingual-mpnet-base-v2']

model = SentenceTransformer(model_names[1])

print('encoding...')
encodings = model.encode(all_words)

scorer = Scorer()
res = scorer(e1=encodings, s1=all_words)
scorer.show(res)

img = plt.imshow(res)
plt.show()

res.to_csv(f'{data_path}/bedsheet_sim.csv', encoding='utf-8-sig')
