import re

import numpy as np


def remove_stopwords(item, usage):
    stop_words = f'当作|当|用|{item}'
    tmp = re.sub(stop_words, '', usage)
    if tmp:
        return re.sub(item, '', usage)
    else:
        return usage


def cos_sim_np(a: np.ndarray, b: np.ndarray):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
