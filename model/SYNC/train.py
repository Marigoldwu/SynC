# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.cluster import KMeans
from model.SYNC.model import SYNC
from utils.data_processor import normalize_adj_torch, similarity, get_importance_of_edge, target_distribution
from utils.evaluation import eva
from utils.result import Result
from utils.utils import get_format_variables, count_parameters


def train(args, data, logger):
    # parameters settings
    params_dict = {"acm": [50, 2e-3, 512, 1],
                   "dblp": [50, 2e-2, 512, 1],
                   "cite": [50, 6e-3, 512, 1],
                   "amap": [50, 1e-4, 512, 1],
                   "uat": [50, 1e-3, 128, 1],
                   "cora": [50, 5e-3, 128, 0],
                   "wisc": [50, 1e-2, 512, 1],
                   "texas": [50, 5e-3, 512, 0]
                   }
    args.hidden_dim = 256
    args.embedding_dim = 16
    args.max_epoch = params_dict[args.dataset_name][0]
    args.lr = params_dict[args.dataset_name][1]
    args.linear_dim = params_dict[args.dataset_name][2]
    args.beta = params_dict[args.dataset_name][3]

    # initialize model
    pretrain_tigae_filename = args.pretrain_save_path + args.dataset_name + ".pkl"
    model = SYNC(args.input_dim, args.hidden_dim, args.embedding_dim,
                 args.linear_dim, args.clusters).to(args.device)
    model.tigae.load_state_dict(torch.load(pretrain_tigae_filename, map_location='cpu'))
    logger.info(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    # load data
    feature = data.feature.to(args.device).float()
    adj_origin = data.adj.to(args.device).float()
    label = data.label
    adj_norm = normalize_adj_torch(adj_origin)

    # initialize clustering centers
    with torch.no_grad():
        _, embedding, _ = model.tigae(feature, adj_norm)
    kmeans = KMeans(n_clusters=args.clusters, n_init=20)
    kmeans.fit_predict(embedding.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(args.device)

    acc_max, embedding2, max_embedding = -1, None, embedding
    max_acc_corresponding_metrics = [0, 0, 0, 0]

    for epoch in range(1, args.max_epoch + 1):
        # Representation Learning Promotes the Generation of Reliable Structure
        with torch.no_grad():
            model.eval()
            A_pred, embedding1, xt1 = model.tigae(feature, adj_norm)
            sx = similarity(xt1)
            # Eq. 5
            A_p = (A_pred + sx) / 2
            # Eq. 6
            A_weight = torch.mul(A_p, adj_origin)
            A_tmp = A_p - A_weight
            row_max_values, _ = torch.max(A_tmp, dim=1)
            mask_row = torch.eq(A_tmp, row_max_values[:, None]).float().to(args.device)
            mask = mask_row + mask_row.t()
            mask[mask >= 1] = 1
            A = adj_origin + mask
            # Eq. 7
            A_p = torch.mul(A_p, A)
            A_p[A_p < 0] = 0
            A_p[A_p > 1] = 1
            A = torch.bernoulli(A_p)
            # Eq. 8
            A = (A + A.T) / 2
            A.fill_diagonal_(1)
            A_label = A
            # Eq. 9
            imp_origin, _ = get_importance_of_edge(A)
            tmp = (A_pred + sx + F.normalize(imp_origin, p=2, dim=1)) / 3
            # Eq. 10
            A = torch.mul(tmp, A_label)
            del tmp, imp_origin, A_p, mask, mask_row
            A = normalize_adj_torch(A)
        
        # Structure Augmentation Promotes the Generation of High-quality Embeddings
        model.train()
        # Eq. 12
        A_cons, embedding2, xt2, q = model(feature, A)
        # Eq. 14
        p = target_distribution(q.data)

        # Eq.15
        re_loss = torch.norm(A_cons - A_label) / args.nodes
        ce_loss = F.cross_entropy(q.log(), p)
        loss = re_loss + args.beta * ce_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            # Eq. 16
            pred = q.data.cpu().numpy().argmax(1)
            acc, nmi, ari, f1 = eva(label, pred)
            if acc > acc_max:
                acc_max = acc
                max_embedding = embedding2
                max_acc_corresponding_metrics = [acc, nmi, ari, f1]
            logger.info(get_format_variables(epoch=f"{epoch:0>3d}", acc=f"{acc:0>.4f}", nmi=f"{nmi:0>.4f}",
                                             ari=f"{ari:0>.4f}", f1=f"{f1:0>.4f}"))

    result = Result(embedding=max_embedding, max_acc_corresponding_metrics=max_acc_corresponding_metrics)
    # Get the network parameters
    logger.info("The total number of parameters is: " + str(count_parameters(model)) + "M(1e6).")
    mem_used = torch.cuda.max_memory_allocated(device=args.device) / 1024 / 1024
    logger.info(f"The max memory allocated to model is: {mem_used:.2f} MB.")
    return result
