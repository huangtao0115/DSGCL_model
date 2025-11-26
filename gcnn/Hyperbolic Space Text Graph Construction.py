
import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import umap

# 常量定义
epsilon = 1e-15


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def minmax_normalize(data):
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    return (data - data_min) / (data_max - data_min + 1e-8)

def hyperbolic_distance_poincare(x, y, epsilon=1e-15):
    norm_diff_squared = torch.sum((x - y) ** 2, dim=2)
    norm_x_squared = torch.sum(x ** 2, dim=2)
    norm_y_squared = torch.sum(y ** 2, dim=1)
    common_denominator = (1 - norm_x_squared) * (1 - norm_y_squared) + epsilon
    return torch.acosh(1 + 2 * norm_diff_squared / common_denominator)

def build_sparse_hyperbolic_adjacency(features, k, device):
    num_nodes = features.shape[0]
    chunk_size = 1000
    adj_sparse = sp.lil_matrix((num_nodes, num_nodes), dtype=np.float32)
    for start in range(0, num_nodes, chunk_size):
        end = min(start + chunk_size, num_nodes)
        chunk = features[start:end].unsqueeze(1)
        dist_chunk = hyperbolic_distance_poincare(chunk, features, epsilon)
        diag_mask = torch.eye(dist_chunk.shape[0], dist_chunk.shape[1], device=device).bool()
        dist_chunk[diag_mask] = torch.inf
        topk_vals, topk_indices = torch.topk(dist_chunk, k=k, dim=1, largest=False)
        sim_chunk = 1 / (topk_vals + epsilon)
        for i in range(chunk.shape[0]):
            global_idx = start + i
            for j in range(k):
                neighbor_idx = topk_indices[i, j].item()
                sim_value = sim_chunk[i, j].item()
                adj_sparse[global_idx, neighbor_idx] = sim_value

    return adj_sparse.tocoo()

def normalize(mx):
    rowsum = np.array(mx.sum(1)).flatten()
    r_inv = np.power(rowsum, -0.5)
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx).dot(r_mat_inv)


name = 'snippets'  # snippets  StackOverflow   ohsumed    TagMyNews   ohsumed   mr   Twitter
k = 25
features = torch.load(f'../glove/embedding_{name}.pt').float()
features_np = features.cpu().detach().numpy()  # snippets

features_np = minmax_normalize(features_np)

reducer = umap.UMAP(
    n_components=30,
    n_neighbors=15,
    min_dist=0.1,
    metric='cosine',
    random_state=42,
    low_memory=True
)
features_reduced = reducer.fit_transform(features_np)
features = torch.from_numpy(features_reduced).float().to(device)

max_norms = torch.max(torch.abs(features), dim=1, keepdim=True)[0]
max_norms[max_norms == 0] = epsilon
features = features / max_norms

adj_sparse = build_sparse_hyperbolic_adjacency(features, k, device)

adj_normalized = normalize(adj_sparse)

adj_dense = adj_normalized.toarray()
adj_tensor = torch.FloatTensor(adj_dense)
torch.save(adj_tensor, f'H_adj_{name}.pt')
























# import scipy.sparse as sp
# from scipy.sparse import coo_matrix
# epsilon = 1e-15
# # #
# def normalize(mx):
#     """Row-normalize sparse matrix"""
#     rowsum = np.array(mx.sum(1))
#     print(1)
#     r_inv = np.power(rowsum, -0.5).flatten()
#     print(2)
#     r_inv[np.isinf(r_inv)] = 0.
#     print(3)
#     r_mat_inv = sp.diags(r_inv)
#     print(4)
#     mx = r_mat_inv.dot(mx).dot(r_mat_inv)
#     return mx
# def hyperbolic_distance_poincare(x, y, epsilon=1e-15):
#     # 计算两个向量之间的欧几里得距离的平方
#     norm_diff_squared = torch.sum((x - y) ** 2, dim=1)  # 直接对一维张量求和
#     # 计算两个向量的范数的平方
#     norm_x_squared = torch.sum(x ** 2, dim=1)
#     norm_y_squared = torch.sum(y ** 2, dim=1)
#     # 计算双曲距离
#     common_denominator = (1 - norm_x_squared) * (1 - norm_y_squared)[:, None] + epsilon  # 添加epsilon避免除以零
#     hyperbolic_dist = torch.acosh(1 + 2 * norm_diff_squared / common_denominator)
#     return hyperbolic_dist
# ##############################     这是在A100上运行的程序   ################################
#
# import torch
# # import geoopt
# # from geoopt.manifolds import PoincareBall
# import numpy as np
#
#
# name = 'Twitter'  # mr, snippets, Twitter
# k = 25  # 每个节点只保留最相似的50个邻居
# # 加载特征并转移到GPU
# features = torch.load(f'../glove/embedding_{name}.pt').float()
#
# # ------------------------- UMAP降维 -------------------------
# features = features.detach().cpu().numpy()  # 分离梯度 + 移动到CPU
# import umap
# # 方案一：Min-Max归一化（缩放到[0,1]范围）
# def minmax_normalize(data):
#     data_min = np.min(data, axis=0)
#     data_max = np.max(data, axis=0)
#     return (data - data_min) / (data_max - data_min + 1e-8)  # 添加极小值防止除以0
#
# features_np = minmax_normalize(features)
#
# reducer = umap.UMAP(
#     n_components=10,
#     n_neighbors=15,
#     min_dist=0.1,
#     metric='cosine',
#     random_state=42
# )
# features = reducer.fit_transform(features_np)
#
#
#
#
# # features = torch.load('../glove/embedding_' + name + '_clean.pt')
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# features = torch.from_numpy(features).float().to(device)
# # 计算每一行特征向量的最大范数
# max_norms = torch.max(torch.abs(features), dim=1, keepdim=True)[0]
# # 避免除以零的情况
# max_norms[max_norms == 0] = epsilon
# # 将每一行的特征除以其最大范数进行归一化
# features = features / max_norms
# # 初始化双曲距离矩阵
# num_points = features.shape[0]
# distance_matrix = torch.zeros((num_points, num_points), device=device)
# # 计算所有点对之间的双曲距离
# diff = features.unsqueeze(1) - features.unsqueeze(0)
# norm_diff_squared = torch.sum(diff ** 2, dim=2)
# norm_x_squared = torch.sum(features ** 2, dim=1, keepdim=True)
# norm_y_squared = norm_x_squared.transpose(1, 0)
# common_denominator = (1 - norm_x_squared) * (1 - norm_y_squared) + epsilon
# hyperbolic_dist = torch.acosh(1 + 2 * norm_diff_squared / common_denominator)
# # 将对角线元素设置为0
# distance_matrix = hyperbolic_dist * (1 - torch.eye(num_points, device=device))
# # 保存相似性矩阵
# torch.save(distance_matrix, 'H_adj_glove_0013_'+name+'.pt')
#
#
#
#
#
# ####################    对H的距离矩阵进行处理     ##############################
#
# hyperbolic_dist = torch.load('H_adj_glove_0013_'+name+'.pt')
# import torch
# print(hyperbolic_dist)
# # 找到最大距离
# # max_dist = torch.max(hyperbolic_dist)
# number=len(hyperbolic_dist)
# # 创建一个新的张量来存储修改后的值
# new_hyperbolic_dist = hyperbolic_dist.clone().detach()
# new_hyperbolic_dist1 = hyperbolic_dist.clone().detach()
# # 计算每一行特征向量的最大范数
# max_norms = torch.max(torch.abs(new_hyperbolic_dist), dim=1, keepdim=True)[0]
# # 避免除以零的情况
# max_norms[max_norms == 0] = 1
# # 将每一行的特征除以其最大范数进行归一化
# new_hyperbolic_dist = new_hyperbolic_dist / max_norms
# print(new_hyperbolic_dist)
# # 将对角线元素设置为1
# for i in range(number):
#     new_hyperbolic_dist[i, i] = 1
# zero_indices = torch.where(new_hyperbolic_dist == 0)
# # 排除对角线元素
# non_diagonal_zero_indices = [(i, j) for i, j in zip(zero_indices[0], zero_indices[1]) if i != j]
# if non_diagonal_zero_indices:
#     print("存在非对角线元素为0，它们的位置是：")
#     for i, j in non_diagonal_zero_indices:
#         new_hyperbolic_dist[i][j] = 1
#         new_hyperbolic_dist[j][i] = 1
# # 计算相似性矩阵
# new_hyperbolic_dist = 1 / new_hyperbolic_dist
# # 将对角线元素设置为0
# for i in range(number):
#     new_hyperbolic_dist[i, i] = 0
# zero_indices = torch.where(new_hyperbolic_dist1 == 0)
# # 排除对角线元素
# non_diagonal_zero_indices = [(i, j) for i, j in zip(zero_indices[0], zero_indices[1]) if i != j]
# if non_diagonal_zero_indices:
#     print("存在非对角线元素为0，它们的位置是：")
#     for i, j in non_diagonal_zero_indices:
#         new_hyperbolic_dist[i][j] = 0
#         new_hyperbolic_dist[j][i] = 0
# else:
#     print("除了对角线元素外，没有其他元素为0")
# # 获取相似性矩阵中的最大值
# max_similarity = torch.max(new_hyperbolic_dist)
# # 将对角线元素设置为最大值的10倍
# for i in range(number):
#     new_hyperbolic_dist[i, i] = max_similarity * 10
# zero_indices = torch.where(new_hyperbolic_dist1 == 0)
# # 排除对角线元素
# non_diagonal_zero_indices = [(i, j) for i, j in zip(zero_indices[0], zero_indices[1]) if i != j]
# if non_diagonal_zero_indices:
#     print("存在非对角线元素为0，它们的位置是：")
#     for i, j in non_diagonal_zero_indices:
#         new_hyperbolic_dist[i][j] = max_similarity * 10
#         new_hyperbolic_dist[j][i] = max_similarity * 10
# else:
#     print("除了对角线元素外，没有其他元素为0")
#
# print(max_similarity)
# # 将张量从CUDA设备（GPU）复制到CPU
# similarity_matrix = new_hyperbolic_dist.cpu()
# # 将PyTorch张量转换为NumPy数组
# similarity_matrix = similarity_matrix.detach().numpy()
# adj_sparse = coo_matrix(similarity_matrix)
# print('开始归一化和数据处理')
# adj = normalize(adj_sparse)
# print('转为torch')
# # 假设 adj 是你的稀疏矩阵
# adj_dense = adj.toarray()  # 将稀疏矩阵转换为密集矩阵
# adj_tensor = torch.FloatTensor(adj_dense)  # 将密集矩阵转换为PyTorch张量
# print('保存图')
# torch.save(adj_tensor, 'H_adj_'+name+'_clean.pt')
# print(adj_tensor)