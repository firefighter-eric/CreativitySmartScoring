import random
from collections import defaultdict
from itertools import chain
from pprint import pprint

import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# %%
df_words = pd.read_excel('data/association_task/最终名词表.xlsx')
words = df_words['词语'].to_list()
n_words = len(words)

# %%
# git lfs clone https://huggingface.co/BAAI/bge-large-zh-v1.5
model = SentenceTransformer(r"H:\models\BAAI\bge-large-zh-v1.5").cuda()
logger.info(f'{model=} loaded')
# %%
embeddings = model.encode(words)
scores = embeddings @ embeddings.T

# %% hist
# from matplotlib import pyplot as plt
# plt.hist(scores_list, bins=10)
# plt.show()
score_list = []
scores_ = scores.tolist()
for i in range(n_words):
    for j in range(i + 1, n_words):
        s = scores_[i][j]
        score_list.append(s)
score_list.sort()
n_scores = len(score_list)
score_bin_dict = {}
for s in tqdm(score_list):
    s = min(s, 1)
    s = max(s, 0)
    min_s = int(s * 10) / 10
    max_s = min_s + 0.1
    key = f'[{min_s:.1f}<{max_s:.1f})'
    score_bin_dict[key] = score_bin_dict.get(key, 0) + 1
pprint(score_bin_dict)

"""
{'[0.0<0.1)': 9,
 '[0.1<0.2)': 18946,
 '[0.2<0.3)': 1264109,
 '[0.3<0.4)': 5314831,
 '[0.4<0.5)': 3429341,
 '[0.5<0.6)': 598773,
 '[0.6<0.7)': 61202,
 '[0.7<0.8)': 8789,
 '[0.8<0.9)': 1479,
 '[0.9<1.0)': 146}
"""


# %%
def describe_upper_matrix(x):
    """
    upper_matrix
    0111
    0011
    0001
    0000
    :param x:
    :return:
    """
    if isinstance(x, np.ndarray):
        x = x.tolist()
    h, w = len(x), len(x[0])
    score_upper = []
    for i in range(h):
        for j in range(i + 1, w):
            score_upper.append(x[i][j])
    average = np.mean(score_upper)
    std = np.std(score_upper)
    # print(f'{h=} {w=} {len(score_upper)=} {average:.4f}±{std:.4f}')
    return {'average': average, 'std': std, 'n': len(score_upper), 'h': h, 'w': w}


total_stat = describe_upper_matrix(scores)
pprint(total_stat)


# %%


def get_clusters(scores: np.ndarray, n_words_per_cluster, n_clusters, min_s, max_s) -> list[dict]:
    index_list = np.where((scores >= min_s) & (scores < max_s))
    index_list = [(x, y) for x, y in zip(index_list[0], index_list[1])]
    index_map = defaultdict(set)
    for x, y in index_list:
        if x == y:
            continue
        index_map[x].add(y)

    output = []
    cluster_index_results = set()
    for i in tqdm(range(n_clusters), desc=f'{n_words_per_cluster=} {min_s=} {max_s=}'):
        while True:
            result = get_one_cluster(index_map=index_map, n_words_per_cluster=n_words_per_cluster)
            if result in cluster_index_results or len(result) < n_words_per_cluster:
                continue

            score_matrix = [[scores[i, j] for i in result] for j in result]
            word = [words[i] for i in result]
            stat = describe_upper_matrix(score_matrix)
            cluster_index_results.add(result)

            output.append({
                'scores': list(chain(*score_matrix)),
                'words': word,
                'score_average': stat['average'],
                'score_std': stat['std'],
            })
            break

    return output


def get_one_cluster(index_map, n_words_per_cluster) -> tuple:
    result = set()
    while len(result) < n_words_per_cluster:
        if not result:
            candidates = list(index_map.keys())
        else:
            # candidates = [_ for _ in index_list if _[0] in result]
            candidates = set(index_map.keys())
            for r in result:
                candidates = candidates & index_map[r]
            candidates = list(candidates)
        if not candidates:
            break
        item = random.choice(candidates)
        result.add(item)
    result = tuple(result)
    return result


n_words_per_cluster = 3
n_clusters = 100
min_s = 0.0
max_s = 0.3
r = get_clusters(scores, n_words_per_cluster, n_clusters, min_s, max_s)
pprint(r)

# %%

random.seed(42)
n_clusters = 100
for n_words_per_cluster in [3, 4, 5]:
    for min_s, max_s in [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]:
        r = get_clusters(scores, n_words_per_cluster, n_clusters, min_s, max_s)
        filename = f'{n_clusters=}{n_words_per_cluster=}{min_s=}{max_s=}'
        columns = ([f'W{i}' for i in range(n_words_per_cluster)] +
                   ['score_average', 'score_std'] +
                   [f'S{i}{j}' for i in range(n_words_per_cluster) for j in range(n_words_per_cluster)])
        df = pd.DataFrame(columns=columns)

        for i in range(n_clusters):
            df.loc[i] = r[i]['words'] + [r[i]['score_average']] + [r[i]['score_std']] + r[i]['scores']
        df.to_csv(f'data/association_task/{filename}.csv', index=False, encoding='utf-8-sig')
