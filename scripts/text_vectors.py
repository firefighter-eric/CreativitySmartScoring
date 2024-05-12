import pickle

import pandas as pd

from css.get_model import get_model
from css.utils import remove_stopwords

input_path = 'data/pku_raw.csv'

# load data
df_in = pd.read_csv(input_path)
df_in = df_in.dropna(how='any')
df_in['Usage'] = df_in.apply(lambda row: remove_stopwords(row['Item'], row['Usage']), axis=1)
usages = df_in['Usage'].tolist()

# model inference
model_name = 'simcse_cyclone'
model = get_model('simcse_cyclone')
sents_vec = model(usages)


with open('outputs/pku_vectors.pkl', 'wb') as fo:
    pickle.dump(sents_vec, fo)
