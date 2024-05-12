import re
from collections import defaultdict

import pandas as pd
from sentence_transformers import util, SentenceTransformer

import envs
from css import metric


def filter_by(item, usage):
    stop_words = f'当作|当|用|{item}'
    tmp = re.sub(stop_words, '', usage)
    if tmp:
        return re.sub(item, '', usage)
    else:
        return usage


def get_cos_sim(model, target, sents):
    print('encoding...')
    sents_vec = model.encode(sents)
    target_vec = model.encode([target])
    cos_sim = util.cos_sim(sents_vec, target_vec).squeeze(-1)
    return cos_sim


def get_pearson_spearson(labels: dict):
    columns = list(labels.keys())
    person = pd.DataFrame(columns=columns)
    spearson = pd.DataFrame(columns=columns)
    for rater1, label1 in labels.items():
        for rater2, label2 in labels.items():
            person.loc[rater1, rater2] = metric.get_pearson_corr(label1, label2)
            spearson.loc[rater1, rater2] = metric.get_spearman_corr(label1, label2)
    return person, spearson


data_path = f'{envs.project_path}/data'
out_path = f'{envs.project_path}/outputs'
out = {}

file_path = f'{data_path}/sjm_long.csv'

df_in = pd.read_csv(file_path)
df_in['用途'] = df_in.apply(lambda row: filter_by(row['item'], row['用途']), axis=1)
df_out = df_in.copy()

data = defaultdict(list)
usage_dict = defaultdict(set)
inappropriate = defaultdict(set)
for i, line in df_in.iterrows():
    item = line['item']
    usage = line['用途']
    score = line['score']
    rID = line['ResponseID']
    sID = line['SubID']
    rater = line['rater']

    usage_dict[item].add(usage)
    data[item].append({
        'usage': usage,
        'usage_len': len(usage),
        'score': score,
        'rID': rID,
        'sID': sID,
        'rater': rater,
    })
    if line['合适'] == 1:
        inappropriate[item].add(usage)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

out = dict()
for item, usages in usage_dict.items():
    usages = list(usages)
    cos_sim = get_cos_sim(model=model, target=item, sents=usages).tolist()
    out[item] = {u: s for u, s in zip(usages, cos_sim)}

df_scores = []
for _, line in df_in.iterrows():
    item = line['item']
    usages = line['用途']
    df_scores.append(out[item][usages])

df_out['sbert_score'] = df_scores
# 加的用途
out = dict()
for item, usages in usage_dict.items():
    usages = list(usages)
    cos_sim = get_cos_sim(model=model, target=item + '的用途', sents=usages).tolist()
    out[item] = {u: s for u, s in zip(usages, cos_sim)}

df_scores = []
for _, line in df_in.iterrows():
    item = line['item']
    usages = line['用途']
    df_scores.append(out[item][usages])

df_out['的用途_sbert_score'] = df_scores
# out

df_out.to_csv(f'{out_path}/sjm_out_1.csv', index=False)


