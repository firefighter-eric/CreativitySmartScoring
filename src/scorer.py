from pandas import DataFrame
from sentence_transformers import util


class Scorer:
    def __init__(self):
        pass

    def __call__(self, s1, e1, s2=None, e2=None) -> DataFrame:
        if not s2:
            s2, e2 = s1, e1
        cosine_scores = util.pytorch_cos_sim(e1, e2)
        df = DataFrame(cosine_scores.tolist(), index=s1, columns=s2)
        return df

    @staticmethod
    def show(df: DataFrame):
        df = df.round(2)
        print(df)
