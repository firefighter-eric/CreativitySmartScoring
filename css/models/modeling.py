import time
from typing import List

import numpy as np
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from torch import nn
from transformers import AutoModel, BertTokenizer

from css.models import bert_whitening_utils as utils


class ModelMixin:
    def __init__(self, model_name_or_path):
        t0 = time.time()
        self.model = None
        self.load(model_name_or_path)
        t1 = time.time()
        logger.info(f'Model loaded in {t1 - t0:.2f} seconds from {model_name_or_path}')

    def load(self, model_name_or_path):
        pass

    def __call__(self, x: list[str]) -> np.ndarray:
        pass


class Word2VectorFastText(ModelMixin):
    def load(self, model_name_or_path):
        from fasttext import FastText
        self.model = FastText.load_model('C:\Projects\CreativitySmartScoring\models\word2vec\cc.zh.300.bin')

    def __call__(self, x: List[str]):
        x = [_.replace(' ', '') for _ in x]
        v = [self.model.get_word_vector(_) for _ in x]
        v = np.vstack(v)
        return v


class Word2VectorGensim(ModelMixin):
    def load(self, model_name_or_path):
        from gensim.models import KeyedVectors
        self.model = KeyedVectors.load_word2vec_format(model_name_or_path, binary=False)

    def __call__(self, x: list[str]):
        v = [self.model.get_vector(_) for _ in x]
        v = np.vstack(v)
        return v


class Sbert(ModelMixin):
    def load(self, model_name_or_path):
        self.model = SentenceTransformer(model_name_or_path, device='cuda')

    def __call__(self, x):
        return self.model.encode(x)


class BertWhitening:
    def __init__(self, sents, pooling='last_avg'):
        self.pooling = pooling
        self.tokenizer, self.model = utils.build_model('bert-base-chinese')
        sents_vec = utils.sents_to_vecs(sents=sents, tokenizer=self.tokenizer, model=self.model,
                                        pooling=self.pooling, max_length=512)
        self.kernel, self.bias = utils.compute_kernel_bias(sents_vec)

    def __call__(self, x):
        x = utils.sents_to_vecs(sents=x, tokenizer=self.tokenizer, model=self.model,
                                pooling=self.pooling, max_length=512)
        x = utils.transform_and_normalize(x, self.kernel, self.bias)
        return x


class SimCSE(nn.Module):
    def __init__(self, model_name_or_path: str, pooling: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name_or_path)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.bert(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]

        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]

        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]

        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]


class SimCSEPipeLine:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('../../data/models/chinese_roberta_wwm_ext_pytorch')
        self.model = SimCSE('../models/chinese_roberta_wwm_ext_pytorch', pooling='cls')
        self.model.cuda().eval()

    def __call__(self, x, batch_size=16):
        length = len(x)
        output = []
        for i in range(0, length, batch_size):
            l, r = i, min(i + batch_size, length)
            batch_x = x[l: r]
            batch_x = self.tokenizer(batch_x, return_tensors='pt', padding=True).to('cuda')
            batch_x = self.model(**batch_x)
            output.append(batch_x)
        output = torch.cat(output, dim=0)
        return output


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model(model_name, *args):
    if model_name == 'word2vec':
        return Word2VectorFastText()
    elif model_name == 'sbert_minilm':
        return Sbert(model_name_or_path='paraphrase-multilingual-MiniLM-L12-v2')
    elif model_name == 'sbert_mpnet':
        return Sbert(model_name_or_path='paraphrase-multilingual-mpnet-base-v2')
    elif model_name == 'simcse_cyclone':
        return Sbert(model_name_or_path='cyclone/simcse-chinese-roberta-wwm-ext')
    elif model_name == 'simcse_uer':
        return Sbert(model_name_or_path='uer/simcse-base-chinese')
    elif model_name == 'bert':
        return Sbert(model_name_or_path='bert-base-chinese')
    elif model_name == 'bert_whitening':
        return BertWhitening(*args)
    elif model_name == 'simcse_':
        return SimCSEPipeLine()
    else:
        return Sbert(model_name_or_path=model_name)


if __name__ == '__main__':
    _m = Word2VectorGensim('data/models/word2vec/tencent-ailab-embedding-zh-d100-v0.2.0-s.txt')
    _v = _m(['woman', 'man'])
