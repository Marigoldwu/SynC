# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.optim import Adam
from module.TIGAE import TIGAE
from utils import kmeans_gpu
from utils.data_processor import normalize_adj_torch, similarity
from utils.evaluation import eva
from utils.result import Result
from utils.utils import get_format_variables
from utils.data_processor import add_random_edges_torch, aug_feature_dropout
import numpy as np


def train(args, data, logger):
    params_dict = {"acm": [50, 1e-3, 0, 512],
                   "dblp": [20, 2e-3, 1, 512],
                   "cite": [20, 2e-3, 1, 512],
                   "amap": [80, 1e-3, 1, 512],
                   "uat": [50, 2e-3, 0, 128],
                   "cora": [80, 2e-3, 0, 128],
                   "wisc": [20, 1e-2, 1, 512],
                   "texas": [20, 5e-3, 1e1, 512]
                   }
    args.hidden_dim = 256
    args.embedding_dim = 16
    args.pretrain_epoch = params_dict[args.dataset_name][0]
    args.pretrain_lr = params_dict[args.dataset_name][1]
    args.epsilon = params_dict[args.dataset_name][2]
    args.linear_dim = params_dict[args.dataset_name][3]
    args.dropout_rate = 0.3


    pretrain_lgae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"

    model = TIGAE(args.input_dim, args.hidden_dim, args.embedding_dim, args.linear_dim).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.pretrain_lr)
    feature = data.feature.to(args.device).float()
    # feature = aug_feature_dropout(feature, args.dropout_rate)
    # np.save("../PyDGC-0.0.1/dataset/acm/acm_feat.npy", feature.cpu().numpy())
    sf = similarity(feature)
    adj_origin = data.adj.to(args.device).float()
    # adj_origin = add_random_edges_torch(adj_origin, b=args.random_edge_rate)

    adj_norm = normalize_adj_torch(adj_origin).float()
    adj_label = adj_origin
    label = data.label

    acc_max, embedding = 0, None
    max_acc_corresponding_metrics = [0, 0, 0, 0]
    for epoch in range(1, args.pretrain_epoch + 1):
        model.train()
        optimizer.zero_grad()
        A_pred, embedding, xt = model(feature, adj_norm)
        sx = similarity(xt)

        loss1 = torch.norm(A_pred - adj_label) / args.nodes
        loss2 = F.cross_entropy(sx, sf)
        loss = loss1 + args.epsilon * loss2

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred, _ = kmeans_gpu.kmeans(embedding, args.clusters)
            acc, nmi, ari, f1 = eva(label, pred.numpy())
            if acc > acc_max:
                acc_max = acc
                max_acc_corresponding_metrics = [acc, nmi, ari, f1]
                torch.save(model.state_dict(), pretrain_lgae_filename)
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))
    result = Result(embedding=embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    return result
