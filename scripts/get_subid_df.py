import os
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd

import envs

if __name__ == '__main__':
    # config
    data_path = f'{envs.project_path}/data'
    out_path = f'{envs.project_path}/outputs'

    parser = ArgumentParser()
    parser.add_argument('--input_file', default='pku_out.csv')
    parser.add_argument('--output_file', default='pku_sub.csv')
    # parser.add_argument('--input_file', default='sjm_out.csv')
    # parser.add_argument('--output_file', default='sjm_sub.csv')
    args = parser.parse_args()
    file_path = os.path.join(out_path, args.input_file)
    out_file_path = os.path.join(out_path, args.output_file)

    # load
    df = pd.read_csv(file_path)
    raters = [_ for _ in df.columns if _.startswith('Rater')]
    # sub_df = df.groupby('SubID')[raters].mean()

    # sub_df.insert(0, 'SubID', sub_df.index)
    # sub_df.to_csv(out_file_path, index=False)
    items = sorted(set(df['Item']))
    sub_ids = sorted(set(df['SubID']))
    data = defaultdict(lambda: defaultdict(list))
    df_dict = df.to_dict('records')
    for line in df_dict:
        sub_id = line['SubID']
        item = line['Item']
        for rater in raters:
            data[f'{rater}_{item}'][sub_id].append(line[rater])

    df_out = pd.DataFrame()
    df_out['SubID'] = sub_ids
    for column, v in data.items():
        _v = [v[_] for _ in sub_ids]
        arr = [sum(_)/len(_) if _ else np.nan for _ in _v]
        df_out[column] = arr

    df_out.to_csv(out_file_path, index=False)
