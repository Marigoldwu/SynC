# -*- coding: utf-8 -*-
import copy
import random
import torch
import numpy as np
from sklearn.preprocessing import normalize


def numpy_to_torch(a, is_sparse=False):
    """
    numpy array to torch tensor

    :param a: the numpy array
    :param is_sparse: is sparse tensor or not
    :return a: torch tensor
    """
    if is_sparse:
        a = torch.sparse.Tensor(a) # type: ignore
    else:
        a = torch.from_numpy(a)
    return a


def torch_to_numpy(t):
    """
    torch tensor to numpy array

    :param t: the torch tensor
    :return t: numpy array
    """
    return t.numpy()


def data_split(full_list, n_sample):
    offset = n_sample
    random.shuffle(full_list)
    len_all = len(full_list)
    index_now = 0
    split_list = []
    while index_now < len_all:
        # 0-2000
        if index_now+offset > len_all:
            split_list.append(full_list[index_now:])
        else:
            split_list.append(full_list[index_now:index_now+offset])
        index_now += offset
    return split_list


def normalize_adj(adj, symmetry=True):
    """
    normalize the adj matrix

    :param adj: input adj matrix
    :param symmetry: symmetry normalize or not
    :return norm_adj: the normalized adj matrix
    """

    # calculate degree matrix and it's inverse matrix
    d = np.diag(adj.sum(0))
    d_inv = np.linalg.inv(d)

    # symmetry normalize: D^{-0.5} A D^{-0.5}
    if symmetry:
        sqrt_d_inv = np.sqrt(d_inv)
        norm_adj = np.matmul(np.matmul(sqrt_d_inv, adj), sqrt_d_inv)

    # non-symmetry normalize: D^{-1} A
    else:
        norm_adj = np.matmul(d_inv, adj)

    return norm_adj


def normalize_adj_torch(adj, symmetry=True):
    """
    Normalize the adjacency matrix.

    :param adj: Input adjacency matrix
    :param symmetry: Symmetry normalize or not
    :return norm_adj: The normalized adjacency matrix
    """
    # Calculate degree matrix and its inverse matrix
    d_inv = torch.diag(1 / torch.sum(adj, dim=1))

    # Symmetry normalize: D^(-0.5) A D^(-0.5)
    if symmetry:
        sqrt_d_inv = torch.sqrt(d_inv)
        norm_adj = torch.matmul(torch.matmul(sqrt_d_inv, adj), sqrt_d_inv)

    # Non-symmetry normalize: D^(-1) A
    else:
        norm_adj = torch.matmul(d_inv, adj)

    return norm_adj


def construct_graph(feat, k=5, metric="euclidean"):
    """
    construct the knn graph for a non-graph dataset

    :param feat: the input feature matrix
    :param k: hyper-parameter of knn
    :param metric: the metric of distance calculation
    - euclidean: euclidean distance
    - cosine: cosine distance
    - heat: heat kernel
    :return knn_graph: the constructed graph
    """

    # euclidean distance, sqrt((x-y)^2)
    if metric == "euclidean" or metric == "heat":
        xy = np.matmul(feat, feat.transpose())
        xx = (feat * feat).sum(1).reshape(-1, 1)
        xx_yy = xx + xx.transpose()
        euclidean_distance = xx_yy - 2 * xy
        euclidean_distance[euclidean_distance < 1e-5] = 0
        distance_matrix = np.sqrt(euclidean_distance)

        # heat kernel, exp^{- euclidean^2/t}
        if metric == "heat":
            distance_matrix = - (distance_matrix ** 2) / 2
            distance_matrix = np.exp(distance_matrix)

    # cosine distance, 1 - cosine similarity
    if metric == "cosine":
        norm_feat = feat / np.sqrt(np.sum(feat ** 2, axis=1)).reshape(-1, 1)
        cosine_distance = 1 - np.matmul(norm_feat, norm_feat.transpose())
        cosine_distance[cosine_distance < 1e-5] = 0
        distance_matrix = cosine_distance

    # top k
    distance_matrix = numpy_to_torch(distance_matrix) # type: ignore
    top_k, index = torch.topk(distance_matrix, k)
    top_k_min = torch.min(top_k, dim=-1).values.unsqueeze(-1).repeat(1, distance_matrix.shape[-1])
    ones = torch.ones_like(distance_matrix)
    zeros = torch.zeros_like(distance_matrix)
    knn_graph = torch.where(torch.ge(distance_matrix, top_k_min), ones, zeros)
    knn_graph = torch_to_numpy(knn_graph)

    return knn_graph


def get_M(adj, t=2):
    """
    calculate the matrix M by the equation:
        M=(B^1 + B^2 + ... + B^t) / t

    :param t: default value is 2
    :param adj: the adjacency matrix
    :return: M
    """
    tran_prob = normalize(adj, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def get_importance_of_edge(adj):
    adj[adj <= 0] = 0
    D = torch.diag(adj.sum(1)).float()
    D = D.to_sparse()  # Convert to sparse tensor for efficiency
    adj = adj.to_sparse()  # Convert to sparse tensor for efficiency

    Ie = torch.sparse.mm(D, adj)  # Sparse matrix multiplication
    Ie = torch.sparse.mm(Ie, D)  # Sparse matrix multiplication
    # Ie = torch.matmul(torch.matmul(D, adj), D)
    Ie = Ie.to_dense()  # Convert back to dense tensor
    Ie_max = torch.max(Ie)
    Ie_min = torch.min(Ie)
    Ie = (Ie - Ie_min) / (Ie_max - Ie_min)
    return Ie, D


def get_importance_of_node(Imp_edge, D):
    Imp_node = torch.diag(Imp_edge.sum(1)) / D
    Imp_node[torch.isinf(Imp_node)] = 1
    Imp_node[torch.isnan(Imp_node)] = 0
    return Imp_node


def similarity(X, distance='cosine'):
    if distance == 'cosine':
        X = X / torch.norm(X, dim=-1, keepdim=True)
        X[torch.isinf(X)] = 1
        X[torch.isnan(X)] = 0
        S = torch.mm(X, X.t())
    elif distance == 'euclidean':
        # 计算基于欧式距离的相似度矩阵
        sum_X = torch.sum(X**2, dim=1, keepdim=True)
        dist_matrix = sum_X + sum_X.T - 2 * torch.matmul(X, X.T)

        # 处理数值稳定性问题（防止负值由于浮点误差）
        dist_matrix = torch.clamp(dist_matrix, min=0.0)

        # 开平方得到欧氏距离
        dist_matrix = torch.sqrt(dist_matrix)
        S = 1.0 / (1.0 + dist_matrix)

    elif distance == 'manhattan':
        n = X.size(0)
        dist_matrix = torch.zeros((n, n), device=X.device)
        for i in range(n):
            dist_matrix[i] = torch.sum(torch.abs(X - X[i]), dim=1)
        S = 1.0 / (1.0 + dist_matrix)
    else:
        X_centered = X - X.mean(dim=1, keepdim=True)
        cov = torch.matmul(X_centered, X_centered.T)
        std = torch.norm(X_centered, dim=1)
        S = cov / (std.unsqueeze(1) * std.unsqueeze(0) + 1e-6)
    return S


def get_imp_structure(adj):
    adj.fill_diagonal_(0)
    imp_edge, D = get_importance_of_edge(adj)
    imp_node = get_importance_of_node(imp_edge, D)
    imp_structure = torch.sigmoid(torch.matmul(torch.matmul(imp_node, imp_edge), imp_node))
    imp_structure = imp_structure + torch.ones_like(adj).cuda()
    return imp_structure


def aug_feature_dropout(input_feat, drop_rate=0.2):
    """
    dropout features for augmentation.
    args:
        input_feat: input features
        drop_rate: dropout rate
    returns:
        aug_input_feat: augmented features
    """
    aug_input_feat = copy.deepcopy(input_feat).squeeze(0)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_rate)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat


def add_random_edges_torch(adj_matrix, b, device='cuda'):
    """
    使用GPU加速给对称邻接矩阵随机添加边
    
    参数:
    adj_matrix (torch.Tensor): N×N的对称邻接矩阵
    b (float): 添加边的比例系数
    device (str): 'cuda'或'cpu'，指定计算设备
    
    返回:
    torch.Tensor: 添加边后的邻接矩阵
    """
    # 确保矩阵在指定设备上
    if adj_matrix.device != torch.device(device):
        adj_matrix = adj_matrix.to(device)
    
    N = adj_matrix.shape[0]
    
    # 计算当前非自环边数（GPU上高效计算）
    current_edges = (adj_matrix.sum() - adj_matrix.diag().sum()) / 2
    current_edges = int(current_edges.item())
    
    # 计算需添加的边数
    a = min(int(current_edges * b), N*(N-1)//2 - current_edges)
    print(f"Current edges: {current_edges}, Total possible edges: {N*(N-1)//2}, ")
    print(f"Adding {a} random edges to the adjacency matrix.")
    if a <= 0:
        return adj_matrix.clone()
    
    # 使用GPU生成候选边索引（向量化实现）
    mask = (torch.triu(torch.ones(N, N, device=device), diagonal=1) * 
            (1 - adj_matrix)).bool()
    candidate_indices = torch.nonzero(mask, as_tuple=True)
    
    # 随机选择边
    selected_idx = torch.randperm(len(candidate_indices[0]), device=device)[:a]
    i_indices = candidate_indices[0][selected_idx]
    j_indices = candidate_indices[1][selected_idx]
    
    # 更新邻接矩阵（并行操作）
    new_adj = adj_matrix.clone()
    new_adj[i_indices, j_indices] = 1
    new_adj[j_indices, i_indices] = 1  # 保持对称性
    
    return new_adj