import fasttext
from fasttext import util

ft = fasttext.load_model('C:\Projects\CreativitySmartScoring\models\word2vec\cc.zh.300.bin')
print(ft.get_dimension())
util.reduce_model(ft, 100)
print(ft.get_dimension())
