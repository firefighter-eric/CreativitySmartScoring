from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class CSSDataSet(Dataset):
    def __init__(self, split='train'):
        data = self.load_data()
        # train, dev = self.split(item_rater_dict, ratio=0.1)

        train = dev = data

        # train = self.data_augment(train)
        data = {'train': train,
                'dev': dev}

        if split == 'train':
            self.data = data['train']
        elif split == 'dev':
            self.data = data['dev']
        print(f'{split} data loaded')

    @staticmethod
    def load_data():
        df = pd.read_csv('/data/床单_raters.csv')
        data = sorted(set(df['用途'].to_list()))
        data = [[_, _] for _ in data]
        return data

    @staticmethod
    def split(data: list, ratio) -> Tuple[list, list]:
        train, test = train_test_split(data, test_size=ratio, random_state=42)
        return train, test

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    d_train = CSSDataSet()
