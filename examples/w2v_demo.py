import numpy as np
from gensim.models import KeyedVectors
from loguru import logger
import pandas as pd

# %%
file_path = 'data/models/tencent-ailab-embedding-zh-d200-v0.2.0-s/tencent-ailab-embedding-zh-d200-v0.2.0-s.txt'

word_vectors = KeyedVectors.load_word2vec_format(file_path, binary=False)
logger.info('model init')

# %%

similarity = word_vectors.similarity('woman', 'man')

# %%
similarity = word_vectors.similarity('中国', '美国')

# %%

df = pd.read_excel('data/word_association/dataT1.xlsx', sheet_name='assocations')

tmp = '桌子'



columns = [f'Table{i}' for i in range(1, 20)]

df = df[columns]


def func1(row: pd.Series, first_token='桌子'):
    row = [first_token] + row.tolist()
    sims = []
    for i in range(len(row) - 1):
        try:
            s = word_vectors.similarity(row[i], row[i + 1])
        except:
            s = np.nan

        s = 1 - s
        sims.append(s)
    return sims

s_columns = [f's_{i+1}' for i in range(len(columns))]
df[s_columns] = df.apply(func1, axis=1, result_type='expand')

df['s_mean'] = df[s_columns].apply(np.mean())
