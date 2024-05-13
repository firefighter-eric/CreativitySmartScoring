from fasttext import util, FastText

model = FastText.load_model('C:\Projects\CreativitySmartScoring\models\word2vec\cc.zh.300.bin')
print(model.get_dimension())
util.reduce_model(model, 100)
print(model.get_dimension())
v = model.get_word_vector('床单')
