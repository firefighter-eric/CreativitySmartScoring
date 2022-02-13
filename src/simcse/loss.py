import torch
from torch import nn, Tensor


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fct = nn.MSELoss()

    def forward(self, x1: Tensor, x2: Tensor, labels: Tensor):
        output = torch.cosine_similarity(x1, x2)
        return self.loss_fct(output, labels.view(-1))
