import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import util
from tqdm import tqdm

from css.models import modeling

tqdm.pandas()

# %%
file_path = 'data/models/word2vec/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt'

model = modeling.Word2VectorGensim(file_path)
# model = modeling.Sbert('paraphrase-multilingual-MiniLM-L12-v2')
logger.info('model init')


# %%


def func1(tokens: list[str]):
    length = len(tokens)

    # encode
    token_vectors = []
    for token in tokens:
        try:
            vector = model([token])
        except:
            vector = np.nan
        token_vectors.append(vector)
    # print(token_vectors)

    # score
    scores = []
    for i in range(length - 1):
        try:
            s = util.cos_sim(token_vectors[i], token_vectors[i + 1])
            s = float(s)
            s = 1 - s
        except:
            s = np.nan
        scores.append(s)
    # print(len(scores))
    return scores


def func2(tokens: list[str]):
    length = len(tokens)

    # encode
    token_vectors = []
    for token in tokens:
        try:
            vector = model([token])
        except:
            vector = None
        token_vectors.append(vector)
    # print(token_vectors)

    # score
    scores = 0
    count = 0
    for i in range(length):
        for j in range(length):
            if token_vectors[i] is None or token_vectors[j] is None:
                continue
            s = util.cos_sim(token_vectors[i], token_vectors[j])
            s = float(s)
            scores += 1 - s
            count += 1

    mean_score = scores / count if count > 0 else np.nan
    return mean_score


# %% exp 1

df = pd.read_excel('data/word_association/dataT1.xlsx', sheet_name='assocations')
first_token = '桌子'
column_prefix = 'Table'
columns = [f'{column_prefix}{i}' for i in range(1, 20)]
df = df[columns]
s_columns = [f's_{i + 1}' for i in range(len(columns))]
df[s_columns] = df.progress_apply(lambda x: func1(tokens=[first_token] + x.tolist()), axis=1, result_type='expand')
df['s_mean'] = df[s_columns].apply(lambda x: np.nanmean(x), axis=1)
df.to_csv('data/word_association/dataT1_exp1.csv', index=False, encoding='utf-8-sig')

# %% exp 2

df = pd.read_excel('data/word_association/dataT1.xlsx', sheet_name='assocations')
column_prefix = 'DS'
columns = [f'{column_prefix}{i}' for i in range(1, 11)]
df = df[columns]
df['s_mean'] = df.progress_apply(lambda x: func2(tokens=x.tolist()), axis=1)
df.to_csv('data/word_association/dataT1_exp2.csv', index=False, encoding='utf-8-sig')
