import random
from pprint import pprint

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# %%
df_words = pd.read_excel('data/association_task/最终名词表.xlsx')
words = df_words['词语'].to_list()

# %%
# git lfs clone https://huggingface.co/BAAI/bge-large-zh-v1.5
model = SentenceTransformer(r"H:\models\BAAI\bge-large-zh-v1.5").cuda()

# %%
embeddings = model.encode(words)
scores = embeddings @ embeddings.T
scores_list = scores.reshape(-1, ).tolist()

# %% hist
# from matplotlib import pyplot as plt
# plt.hist(scores_list, bins=10)
# plt.show()

scores_list.sort()
n_scores = len(scores_list)
score_bin_dict = {}
for s in tqdm(scores_list):
    s = min(s, 1)
    s = max(s, 0)
    min_s = int(s * 10) / 10
    max_s = min_s + 0.1
    key = f'[{min_s:.1f}<{max_s:.1f})'
    score_bin_dict[key] = score_bin_dict.get(key, 0) + 1
pprint(score_bin_dict)

"""
{'[0.0<0.1)': 18,
 '[0.1<0.2)': 37892,
 '[0.2<0.3)': 2528218,
 '[0.3<0.4)': 10629662,
 '[0.4<0.5)': 6858682,
 '[0.5<0.6)': 1197546,
 '[0.6<0.7)': 122404,
 '[0.7<0.8)': 17578,
 '[0.8<0.9)': 2958,
 '[0.9<1.0)': 2200,
 '[1.0<1.1)': 2718}
"""


# %%


def get_clusters(scores: np.ndarray, n_words_per_cluster, n_clusters, min_s, max_s) -> list[dict]:
    index_list = np.where((scores >= min_s) & (scores < max_s))
    index_list = [(x, y) for x, y in zip(index_list[0], index_list[1])]
    cluster_index_results = set()
    cluster_scores = []
    cluster_words = []
    for i in tqdm(range(n_clusters)):
        while True:
            result = get_one_cluster(index_list=index_list, n_words_per_cluster=n_words_per_cluster)
            if result in cluster_index_results:
                continue

            score = [scores[i, j] for i in result for j in result]
            word = [words[i] for i in result]
            cluster_index_results.add(result)
            cluster_scores.append(score)
            cluster_words.append(word)
            break

    output = []
    for i in range(n_clusters):
        output.append({
            'score': cluster_scores[i],
            'words': cluster_words[i]
        })
    return output


def get_one_cluster(index_list, n_words_per_cluster) -> tuple:
    result = set()
    while len(result) < n_words_per_cluster:
        if not result:
            candidates = index_list
        else:
            candidates = [_ for _ in index_list if _[0] in result]
        if not candidates:
            break
        item = random.choice(candidates)
        result.add(item[0])
        result.add(item[1])
    result = tuple(result)
    return result


n_words_per_cluster = 3
n_clusters = 100
min_s = 0.0
max_s = 0.3
r = get_clusters(scores, n_words_per_cluster, n_clusters, min_s, max_s)
pprint(r)

# %%

n_clusters = 100
for n_words_per_cluster in [4, 5]:
    for min_s, max_s in [(0.0, 0.3), (0.3, 0.6), (0.6, 1.0)]:
        r = get_clusters(scores, n_words_per_cluster, n_clusters, min_s, max_s)
        filename = f'{n_clusters=}{n_words_per_cluster=}{min_s=}{max_s=}'
        columns = ([f'W{i}' for i in range(n_words_per_cluster)] +
                   [f'S{i}{j}' for i in range(n_words_per_cluster) for j in range(n_words_per_cluster)])
        df = pd.DataFrame(columns=columns)

        for i in range(n_clusters):
            df.loc[i] = r[i]['words'] + r[i]['score']
        df.to_csv(f'data/association_task/{filename}.csv', index=False, encoding='utf-8-sig')
