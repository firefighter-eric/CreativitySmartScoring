import pytorch_lightning as pl
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from css.models.simcse.args import TrainArgs, ModelArgs
from css.models.simcse.modeling import Encoder
from css import metric


class CSETask(pl.LightningModule):
    def __init__(self, pretrained=False):
        super(CSETask, self).__init__()
        self.encoder = Encoder(ModelArgs, pretrained)

        self.args = self.encoder.backbone.config
        if TrainArgs.mlm:
            self.mlm_head = BertLMPredictionHead(self.args)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self._lambda = 0.1

    def forward(self, s1, s2, label, mlm_label1, mlm_label2):
        e1 = self.encoder(s1)
        e2 = self.encoder(s2)

        z1, z2 = e1[1], e2[1]
        # z1 = self.gather(z1)
        # z2 = self.gather(z2)
        similarity = cos_sim(z1, z2) * 20

        # print(similarity.size())
        label = torch.arange(similarity.size(0), dtype=torch.long).cuda()
        loss = self.loss_func(similarity, label)

        if TrainArgs.mlm:
            mlm_x1 = self.mlm_head(e1[0]).permute(0, 2, 1)
            mlm_x2 = self.mlm_head(e2[0]).permute(0, 2, 1)
            mlm_loss = self.loss_func(mlm_x1, mlm_label1) + self.loss_func(mlm_x2, mlm_label2)
            loss += self._lambda * mlm_loss
        return z1, z2, similarity, loss

    @staticmethod
    def gather(x):
        # Dummy vectors for allgather
        x_list = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        # Allgather
        dist.all_gather(tensor_list=x_list, tensor=x.contiguous())
        # Since allgather results do not have gradients, we replace the
        # current process's corresponding embeddings with original tensors
        x_list[dist.get_rank()] = x
        # Get full batch embeddings: (bs x N, hidden)
        x = torch.cat(x_list, 0)
        return x

    def configure_optimizers(self):
        def lr_foo(epoch):
            if epoch < TrainArgs.warm_up_epochs:
                lr_scale = 0.1 ** (TrainArgs.warm_up_epochs - epoch)  # warm up lr
            else:
                lr_scale = 0.95 ** epoch
            return lr_scale

        optimizer = torch.optim.Adam(self.parameters(), lr=TrainArgs.lr, weight_decay=TrainArgs.weight_decay)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_foo)
        return [optimizer], [scheduler]

    def training_step(self, train_batch, batch_idx):
        _, _, _, loss = self(*train_batch)
        self.log('train/loss', loss)
        self.log('lr', self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])
        return loss

    def validation_step(self, val_batch, batch_idx):
        _, _, similarity, loss = self(*val_batch)
        self.log('val/loss', loss)
        L = similarity.size(0)

        similarity_flatten = (similarity / 20).view(-1).tolist()
        label_flatten = torch.eye(L).view(-1).tolist()
        spearman_corr = metric.get_spearman_corr(similarity_flatten, label_flatten)
        self.log('val/spearman_corr', spearman_corr)
        return loss


def cos_sim(a, b):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))