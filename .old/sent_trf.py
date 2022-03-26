from typing import List
import pandas
from pandas import DataFrame
from sentence_transformers import SentenceTransformer, util

model_names = ['paraphrase-multilingual-MiniLM-L12-v2',
               'paraphrase-multilingual-mpnet-base-v2']

if __name__ == '__main__':
    # _s = ['砖块', '用来打人', '打人', '造房子', 'brick', 'dog', '狗']
    _s = ['砖块', '馒头', '石头', '用来打人', '用[MASK]打人', '用馒头打人', '用砖块打人', '造房子', '用砖块敲人', '用砖块吃人']
    _m = SentenceTransformer(model_names[1])
    _e = _m.encode(_s)

    from scorer import Scorer
    scorer = Scorer()
    res = scorer(_s, _e)
    scorer.show(res)
