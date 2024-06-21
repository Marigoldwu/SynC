# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Parameter

from module.TIGAE import TIGAE


class SYNC(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, linear_dim, clusters, v=1):
        super(SYNC, self).__init__()
        self.tigae = TIGAE(input_dim, hidden_dim, embedding_dim, linear_dim)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(clusters, embedding_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        A_pred, embedding, x_d = self.tigae(x, adj)
        # Eq. 13
        q = 1.0 / (1.0 + torch.sum(torch.pow(embedding.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return A_pred, embedding, x_d, q
