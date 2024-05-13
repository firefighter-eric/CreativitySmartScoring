from collections import defaultdict
from typing import List

import pandas as pd

# config path
# in_path = 'data/coco_reversedModel_PKU.xlsx')
# out_path = 'outputs/pku_topk_output.csv')
in_path = 'data/coco_reversedModel_SJM.xlsx'
out_path = 'outputs/sjm_topk_output.csv'

# load data
df: pd.DataFrame = pd.read_excel(in_path)
df = df[df['适宜性'] == '合适']

Items = sorted(df['Item'].unique())
print(df.columns)
# Raters = [_ for _ in df.columns if _.startswith('Rater')]
Raters = list(df.columns)[7:]


# process
def get_mean_arr(arr: List[float]):
    def mean(_arr):
        return sum(_arr) / len(_arr)

    arr = sorted(arr, reverse=True)
    arr = arr[:3]
    mean_arr = []
    if len(arr) == 1:
        mean_arr = [arr[0]] * 3
    if len(arr) == 2:
        mean_arr = [arr[0], mean(arr), mean(arr)]
    if len(arr) == 3:
        mean_arr = [arr[0], mean(arr[:2]), mean(arr)]
    return mean_arr


data = defaultdict(lambda: defaultdict(list))
for i, line in df.iterrows():
    subid = line['SubID']
    item = line['Item']
    for rater in Raters:
        score = line[rater]
        data[subid][f'{rater}_{item}'].append(score)

data_out = defaultdict(lambda: defaultdict(float))
for subid, score_dict in data.items():
    for rater_item, scores in score_dict.items():
        scores = get_mean_arr(scores)
        for i, score in enumerate(scores):
            data_out[subid][f'{rater_item}_Top{i + 1}'] = score

df_out = pd.DataFrame()

for subid, score_dict in data_out.items():
    _out = {'SubID': int(subid)}
    _out.update(score_dict)
    df_out = df_out.append(_out, ignore_index=True)

df_out['SubID'] = df_out['SubID'].astype(int)
df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
print(df_out)
