import os.path
from dataclasses import dataclass


@dataclass
class DataArgs:
    model_path = 'hfl/chinese-macbert-base'


@dataclass
class ModelArgs:
    # model_path = 'hfl/chinese-macbert-base'
    # model_path = 'hfl/chinese-macbert-large'
    # model_path = 'hfl/chinese-bert-wwm-ext'
    # model_path = 'voidful/albert_chinese_tiny'
    model_path = 'bert-base-chinese'
    # model_path = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
    # model_path = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    dropout = 0.15


@dataclass
class TrainArgs:
    train_batch_size = 128
    dev_batch_size = train_batch_size * 1
    val_check_interval = 1000

    epochs = 100
    precision = 16
    warm_up_epochs = 1
    lr = 1e-5
    weight_decay = 0
    # weight_decay = 1e-6
    model_output_path = os.path.join('C:\Projects\CreativitySmartScoring\models')

    mlm = False


@dataclass
class LargeTrainArgs(TrainArgs):
    train_batch_size = 6
    mlm = False
