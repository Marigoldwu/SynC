# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from module.GCN import GCN


class TIGAE(nn.Module):
    def __init__(self, input_dim, hidden_size, embedding_dim, linear_dim):
        super(TIGAE, self).__init__()
        self.linear1 = Linear(input_dim, linear_dim)
        self.gcn1 = GCN(linear_dim, hidden_size)
        self.gcn2 = GCN(hidden_size, embedding_dim, activeType='no')

    def forward(self, x, adj):
        xt = self.linear1(x)
        h1 = self.gcn1(xt, adj)
        h2 = self.gcn2(h1, adj)
        embedding = F.normalize(h2, p=2, dim=1)
        A_pred = dot_product_decode(embedding)
        return A_pred, embedding, xt


def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
    return A_pred
