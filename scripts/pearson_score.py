import os.path
from argparse import ArgumentParser
from collections import defaultdict

import pandas as pd

import envs
from css import metric


def get_pearson_spearson(labels: dict):
    columns = list(labels.keys())
    pearson = pd.DataFrame(columns=columns)
    spearman = pd.DataFrame(columns=columns)

    for rater1, score1 in labels.items():
        for rater2, score2 in labels.items():
            pearson.loc[rater1, rater2] = metric.get_pearson_corr(score1, score2)
            spearman.loc[rater1, rater2] = metric.get_spearman_corr(score1, score2)
    pearson = pearson.abs()
    spearman = spearman.abs()
    pearson['mean'] = pearson[['Rater1', 'Rater2', 'Rater3', 'Rater4']].mean(axis=1)
    spearman['mean'] = spearman[['Rater1', 'Rater2', 'Rater3', 'Rater4']].mean(axis=1)
    return pearson, spearman


def get_args():
    parser = ArgumentParser()
    # parser.add_argument('--input_file', default='pku_out.csv')
    parser.add_argument('--input_file', default='sjm_out.csv')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # config
    data_path = f'{envs.project_path}/data'
    out_path = f'{envs.project_path}/outputs'
    args = get_args()
    file_path = os.path.join(out_path, args.input_file)
    out = {}

    # data
    df_in = pd.read_csv(file_path)
    df_in = df_in.dropna(how='all')
    rater_list = [_ for _ in df_in.columns if _.startswith('Rater')]
    df_in.dropna(axis=0, how='any', inplace=True)
    df_in[rater_list] = df_in[rater_list].astype('float')

    item_list = list(set(df_in['Item'].tolist()))

    data = defaultdict(lambda: defaultdict(list))
    usage_dict = defaultdict(set)
    inappropriate = defaultdict(set)
    for i, line in df_in.iterrows():
        _item = line['Item']
        _usage = line['Usage']
        _rID = line['ResponseID']
        _sID = line['SubID']
        _raters = []
        for _rater in rater_list:
            score = line[_rater]

            data[_item][_rater].append({
                'usage': _usage,
                'usage_len': len(_usage),
                'rID': _rID,
                'sID': _sID,
                'score': score,
            })

    scores = {item: {rater: [_['score'] for _ in v]
                     for rater, v in d.items()}
              for item, d in data.items()}

    # pearson, spearman
    pearson_dict, spearman_dict = {}, {}
    for item, score in scores.items():
        pearson, spearman = get_pearson_spearson(score)
        pearson_dict[item] = pearson
        spearman_dict[item] = spearman

