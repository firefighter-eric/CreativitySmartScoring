from collections import defaultdict

import pandas as pd
from sentence_transformers import SentenceTransformer, util

import envs
from css import metric


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(axis=0, how='any')
    # df = df.fillna(0)
    sents = df['用途']
    labels = df['人工得分']

    print(f'{len(sents)=}')
    sent_set = set(sents)
    print(f'{len(sent_set)=}')

    d = defaultdict(list)
    for sent, label in zip(sents, labels):
        d[sent].append(label)

    sent_label = {}
    for sent, labels in d.items():
        _min = min(labels)
        _max = max(labels)
        _mean = sum(labels) / len(labels)

        sent_label[sent] = _mean
    return sent_label


def load_data_2(path):
    df = pd.read_csv(path)
    df = df.dropna(axis=0, how='any')
    raters = [_ for _ in df.columns if _.startswith('Rater')]

    d = defaultdict(lambda: defaultdict(list))
    for i, row in df.iterrows():
        sent = row['用途']
        for rater in raters:
            d[sent][rater].append(row[rater])

    out = defaultdict(list)
    for sent, raters in d.items():
        out['sent'].append(sent)
        for rater, labels in raters.items():
            out[rater].append(sum(labels) / len(labels))
    return out


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
out = {}
# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


for item in ['床单', '拖鞋', '牙刷', '筷子'][:1]:
    file_path = f'{data_path}/{item}_raters.csv'
    data = load_data_2(file_path)
    cos_sim = get_cos_sim(model=model, target=item, sents=data['sent'])

    labels = {'model': cos_sim}
    labels.update({k: v for k, v in data.items() if k.startswith('Rater')})
    person, spearson = get_pearson_spearson(labels)
    # out[item] = {'pearson': pearson,
    #              'spearman': spearman}


