import torch
from torch import nn, Tensor
from transformers import BertModel, BertTokenizerFast


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.model = BertModel.from_pretrained('bert-base-chinese')

    def forward(self, x):
        x = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)

        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        attention_mask = (input_ids != 101) * (input_ids != 102) * (input_ids != 103) * attention_mask
        attention_mask.int()
        print(attention_mask)
        x = self.model(**x)
        x = self.mean_pooling(x, attention_mask)
        return x

    @staticmethod
    def mean_pooling(model_output, mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class MaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')
        self.model = BertModel.from_pretrained('bert-base-chinese')

    def forward(self, x):
        x = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)

        # attention_mask = x['attention_mask']
        mask = self.get_mask_of_mask(x['input_ids'])
        # print(mask)
        x = self.model(**x)
        x = self.mean_pooling(x, mask)
        return x

    @staticmethod
    def get_mask_of_mask(x: Tensor):
        return (x == 103).int()

    @staticmethod
    def mean_pooling(model_output, mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # print(input_mask_expanded, input_mask_expanded.size())
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


if __name__ == '__main__':
    s1 = ['我', '汤姆', '冰激凌', '大象', '路灯', '走']
    s2 = ['[MASK]在吃冰激凌']

    # s1 = ['[MASK][MASK][MASK][MASK]激[MASK]']
    # s2 = ['我在吃冰[MASK]凌']

    base_model = BaseModel()
    e1 = base_model(s1)

    mask_model = MaskModel()
    e2 = mask_model(s2)

    from scorer import Scorer

    scorer = Scorer()
    _s = scorer(s1, e1, s2, e2)
