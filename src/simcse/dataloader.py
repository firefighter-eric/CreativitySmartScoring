import random

import torch
from torch.utils.data import DataLoader

from src.simcse.args import ModelArgs
from src.simcse.dataset import CSSDataSet
from src.simcse.modeling import Tokenizer

train_dataset = CSSDataSet(split='train')
dev_dataset = CSSDataSet(split='dev')

tokenizer = Tokenizer(ModelArgs.model_path)

DROPOUT = 0.3
stop_sign = {101, 102, 103}


def get_mlm_label(tokenized_sent):
    tokenized_sent = {k: v for k, v in tokenized_sent.items()}
    batch_size, seq_length = tokenized_sent['input_ids'].size()
    mlm_label = torch.ones(size=(batch_size, seq_length), dtype=torch.long)
    for b in range(batch_size):
        for s in range(seq_length):
            if tokenized_sent['input_ids'][b, s] not in stop_sign and random.random() < DROPOUT:
                mlm_label[b, s] = tokenized_sent['input_ids'][b, s]
                tokenized_sent['input_ids'][b, s] = 103  # [MASK]
            else:
                mlm_label[b, s] = -100

    return tokenized_sent, mlm_label


def collate_fn(batch):
    L = len(batch)
    s1, s2 = zip(*batch)
    s1, s2 = list(s1), list(s2)
    s1 = tokenizer(s1)
    s2 = tokenizer(s2)
    s1, mlm_label1 = get_mlm_label(s1)
    s2, mlm_label2 = get_mlm_label(s2)
    l = torch.arange(0, L, dtype=torch.long)
    # return s1, s2, l
    return s1, s2, l, mlm_label1, mlm_label2


def get_dataloader(train_args):
    print(f'{train_args.train_batch_size}')
    train_dataloader = DataLoader(train_dataset, train_args.train_batch_size, shuffle=True, num_workers=1,
                                  pin_memory=True, drop_last=True, persistent_workers=True, collate_fn=collate_fn)
    dev_dataloader = DataLoader(dev_dataset, train_args.dev_batch_size, shuffle=False, num_workers=1,
                                pin_memory=True, drop_last=True, persistent_workers=True, collate_fn=collate_fn)
    return train_dataloader, dev_dataloader
