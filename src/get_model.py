import numpy as np
from fasttext import FastText
from sentence_transformers import SentenceTransformer

from src import bert_whitening_utils as utils


class Word2vec:
    def __init__(self):
        self.model = FastText.load_model('C:\Projects\CreativitySmartScoring\models\word2vec\cc.zh.300.bin')

    def __call__(self, x):
        v = [self.model.get_word_vector(_) for _ in x]
        v = np.vstack(v)
        return v


class Sbert:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def __call__(self, x):
        return self.model.encode(x)


class BertWhitening:
    tokenizer, model = utils.build_model('bert-base-chinese')

    def __init__(self, sents, pooling='last_avg'):
        self.pooling = pooling
        sents_vec = utils.sents_to_vecs(sents=sents, tokenizer=self.tokenizer, model=self.model,
                                        pooling=self.pooling, max_length=512)
        self.kernel, self.bias = utils.compute_kernel_bias(sents_vec)

    def __call__(self, x):
        x = utils.sents_to_vecs(sents=x, tokenizer=self.tokenizer, model=self.model,
                                pooling=self.pooling, max_length=512)
        x = utils.transform_and_normalize(x, self.kernel, self.bias)
        return x


def get_model(model_name, *args):
    if model_name == 'word2vec':
        return Word2vec()
    elif model_name == 'sbert':
        return Sbert()
    elif model_name == 'bert_whitening':
        return BertWhitening(*args)
