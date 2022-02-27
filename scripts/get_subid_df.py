import os
from argparse import ArgumentParser

import pandas as pd

import envs

if __name__ == '__main__':
    # config
    data_path = f'{envs.project_path}/data'
    out_path = f'{envs.project_path}/outputs'

    parser = ArgumentParser()
    parser.add_argument('--input_file', default='pku_out_1.csv')
    parser.add_argument('--output_file', default='pku_sub_1.csv')
    args = parser.parse_args()
    file_path = os.path.join(out_path, args.input_file)
    out_file_path = os.path.join(out_path, args.output_file)

    # load
    df = pd.read_csv(file_path)
    raters = [_ for _ in df.columns if _.startswith('Rater')]
    sub_df = df.groupby('SubID')[raters].mean()
    sub_df.insert(0, 'SubID', sub_df.index)
    sub_df.to_csv(out_file_path, index=False)
