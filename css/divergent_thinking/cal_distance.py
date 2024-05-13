from os.path import join

import pandas as pd
from sentence_transformers import util
from tqdm import tqdm

from css.models.get_model import get_model


def mean_distance(arr):
    cos_sim = util.cos_sim(arr, arr)
    batch_size = cos_sim.size(0)

    s = 0
    c = 0
    for y in range(batch_size):
        for x in range(batch_size):
            if y < x:
                s += cos_sim[y, x]
                c += 1

    return float(s / c)


in_path = 'data/sjm_DAT_260.csv'
df = pd.read_csv(in_path)

model_names = ['word2vec', 'bert', 'sbert_mpnet', 'sbert_minilm', 'simcse_cyclone']
features = ['C', 'Z']

for model_name in model_names:
    print(model_name)
    model = get_model(model_name)
    for f in features:
        dists = []
        columns = [f'{f}{i}' for i in range(1, 11)]
        for row in tqdm(df.to_dict('records')):
            sents = [row[c] for c in columns]
            embeddings = model(sents)
            dis = mean_distance(embeddings)
            dists.append(dis)
        df[f'{f}_{model_name}_mean'] = dists

df.to_csv(join(project_path, 'outputs', 'sjm_dist.csv'), index=False)
