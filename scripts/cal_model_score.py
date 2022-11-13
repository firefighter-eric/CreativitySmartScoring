import os.path
from argparse import ArgumentParser
from collections import defaultdict
from os.path import dirname

import pandas as pd
from sentence_transformers import util

from src import metric
from src.get_model import get_model
from src.utils import remove_stopwords


def get_cos_sim(model, target, sents):
    print('encoding...')
    sents_vec = model(sents)
    target_vec = model([target])
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


def process(model_name, tag=''):
    print(f'{model_name}')
    model = None
    if 'whitening' not in model_name:
        model = get_model(model_name)
    out = dict()
    for item, usages in usage_dict.items():
        usages = list(usages)
        if 'whitening' in model_name:
            model = get_model(model_name, usages)
        cos_sim = get_cos_sim(model=model, target=item + tag, sents=usages).tolist()
        out[item] = {u: s for u, s in zip(usages, cos_sim)}

    df_scores = []
    for _, line in df_in.iterrows():
        item = line['Item']
        usage = line['Usage']
        df_scores.append(out[item][usage])

    column_name = f'Rater_{model_name}'
    column_name += f'_{tag}' if tag else ''
    df_out[column_name] = df_scores


def get_args():
    parser = ArgumentParser()
    # parser.add_argument('--input_path', default='data/pku_raw.csv')
    # parser.add_argument('--output_path', default='outputs/pku_out.csv')
    parser.add_argument('--input_path', default='data/sjm_raw.csv')
    parser.add_argument('--output_path', default='outputs/sjm_out.csv')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # config
    args = get_args()
    input_path = args.input_path
    output_path = args.output_path

    # load data
    out = {}
    df_in = pd.read_csv(input_path)
    df_in = df_in.dropna(how='any')
    df_in['Usage'] = df_in.apply(lambda row: remove_stopwords(row['Item'], row['Usage']), axis=1)
    rater_list = [_ for _ in df_in.columns if _.startswith('Rater')]
    df_out = df_in.copy()

    data = defaultdict(list)
    usage_dict = defaultdict(set)
    for i, line in df_in.iterrows():
        _item = line['Item']
        _usage = line['Usage']
        _rID = line['ResponseID']
        _sID = line['SubID']
        _raters = []
        for rater in rater_list:
            _raters.append(line[rater])

        usage_dict[_item].add(_usage)
        data[_item].append({
            'usage': _usage,
            'usage_len': len(_usage),
            'rID': _rID,
            'sID': _sID,
            'raters': _raters,
        })

    # model process
    # model_name = ['word2vec', 'bert', 'bert_whitening', 'sbert_mpnet', 'sbert_minilm', 'simcse_cyclone']
    # model_name = ['simcse_uer']
    model_name = ['simcse_cyclone']
    for _ in model_name:
        process(_)
    for _ in model_name:
        process(_, tag='的用途')

    # output
    os.makedirs(dirname(output_path), exist_ok=True)
    # df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
