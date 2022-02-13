from collections import defaultdict

import pandas as pd

import envs
from sentence_transformers import util, SentenceTransformer
from src import metric, utils
from src.simcse.task import CSETask


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(axis=0, how='any')
    # df = df.fillna(0)
    sents = df['用途']
    labels = df['人工得分']

    print(f'{len(sents)=}')
    sent_set = set(sents)
    print(f'{len(sent_set)=}')

    d = defaultdict(list)
    for sent, label in zip(sents, labels):
        d[sent].append(label)

    sent_label = {}
    for sent, labels in d.items():
        _min = min(labels)
        _max = max(labels)
        _mean = sum(labels) / len(labels)

        sent_label[sent] = _mean
    return sent_label


def load_data_2(path):
    df = pd.read_csv(path)
    df = df.dropna(axis=0, how='any')
    raters = [_ for _ in df.columns if _.startswith('Rater')]

    d = defaultdict(lambda: defaultdict(list))
    for i, row in df.iterrows():
        sent = row['用途']
        for rater in raters:
            d[sent][rater].append(row[rater])

    out = defaultdict(list)
    for sent, raters in d.items():
        out['sent'].append(sent)
        for rater, labels in raters.items():
            out[rater].append(sum(labels) / len(labels))
    return out


def get_cos_sim(model, target, sents):
    print('encoding...')
    sents_vec = model.encode(sents)
    target_vec = model.encode([target])
    cos_sim = util.cos_sim(sents_vec, target_vec).squeeze(-1)
    return cos_sim


def get_pearson_spearson(labels: dict):
    columns = list(labels.keys())
    person = pd.DataFrame(columns=columns)
    spearson = pd.DataFrame(columns=columns)
    for rater1, label1 in labels.items():
        for rater2, label2 in labels.items():
            person.loc[rater1, rater2] = metric.get_pearson_corr(label1, label2)
            spearson.loc[rater1, rater2] = metric.get_spearman_corr(label1, label2)
    return person, spearson


data_path = f'{envs.project_path}/data'
out = {}

# tokenizer, model = utils.build_model('bert-base-chinese')
# task = CSETask.load_from_checkpoint(r'C:\Projects\CreativitySmartScoring\models\lightning_logs\version_4\checkpoints\epoch=9-step=689.ckpt')
# model = task.encoder.backbone.to('cuda')


file_path = f'{data_path}/sjm_long.csv'

raw = pd.read_csv(file_path)

data = defaultdict(list)
usage_set = defaultdict(set)
inappropriate = defaultdict(set)
for i, line in raw.iterrows():
    item = line['item']
    usage = line['用途']
    score = line['score']
    rID = line['ResponseID']
    sID = line['SubID']
    rater = line['rater']

    usage_set[item].add(usage)
    data[item].append({
        'usage': usage,
        'usage_len': len(usage),
        'score': score,
        'rID': rID,
        'sID': sID,
        'rater': rater,
    })
    if line['合适'] == 1:
        inappropriate[item].add(usage)


# target = [item]
# sents_vec = utils.sents_to_vecs(sents=sents, tokenizer=tokenizer, model=model, pooling=pooling, max_length=512)
# target_vec = utils.sents_to_vecs(sents=target, tokenizer=tokenizer, model=model, pooling=pooling, max_length=512)
# # kernel, bias = utils.compute_kernel_bias(sents_vec)
# # # kernel = kernel[:, :250]
# # sents_vec = utils.transform_and_normalize(sents_vec, kernel, bias)
# # target_vec = utils.transform_and_normalize(target_vec, kernel, bias)
# cos_sim = util.cos_sim(sents_vec, target_vec).squeeze(-1).tolist()
#
# labels = {'model': cos_sim}
# labels.update({k: v for k, v in item_rater_dict.items() if k.startswith('Rater')})
# pearson, spearman = get_pearson_spearson(labels)
# # out[item] = {'pearson': pearson,
# #              'spearman': spearman}


model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

for item in ['床单', '拖鞋', '牙刷', '筷子'][:1]:
    usage = list(usage_set[item])
    cos_sim = get_cos_sim(model=model, target=item, sents=usage)

    # labels = {'model': cos_sim}
    # labels.update({k: v for k, v in item_rater_dict.items() if k.startswith('Rater')})

    person, spearson = get_pearson_spearson(labels)
    out[item] = {'pearson': pearson,
                 'spearman': spearson}
