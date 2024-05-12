import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig


class Encoder(nn.Module):
    def __init__(self, args=None, load_pretrained=False, path=''):
        super().__init__()
        if load_pretrained:
            if not path:
                path = args.model_path
            self.backbone = AutoModel.from_pretrained(path)
        else:
            config = AutoConfig.from_pretrained(args.model_path)
            self.backbone = AutoModel.from_config(config)

        # hidden_size = self.backbone.config.hidden_size
        # self.mlp = nn.Linear(hidden_size, hidden_size)
        # freeze_module(self.backbone)

    def forward(self, x):
        attention_mask = x['attention_mask']
        x = self.backbone(**x)
        # x = x[1]
        # x = self.mlp(x)
        # x = self.mean_pooling(x, attention_mask)
        return x

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        # Mean Pooling - Take attention mask into account for correct averaging
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class Tokenizer:
    def __init__(self, path):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.model_max_length = 512

    def __call__(self, x):
        return self.tokenizer(x, padding=True, truncation=True, return_tensors='pt')


def freeze_module(net: nn.Module):
    for para in net.parameters():
        para.requires_grad = False


def unfreeze_module(net: nn.Module):
    for para in net.parameters():
        para.requires_grad = True
