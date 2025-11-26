import torch
import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix, lil_matrix
from umap import UMAP
import gc


k = 25 # 0 5 10 15 20 30 35 40 45 50 100


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx).dot(r_mat_inv)

name = 'snippets'  # mr, snippets, Twitter   StackOverflow  ohsumed   TagMyNews     Twitter
features = torch.load(f'../glove/embedding_{name}.pt').float()

max_norms = torch.max(torch.abs(features), dim=1, keepdim=True)[0]
max_norms[max_norms == 0] = 1
features = features / max_norms

# 2. UMAP
features_np = features.cpu().detach().numpy().copy()
del features
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

umap_reducer = UMAP(
    n_components=30,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    random_state=42,
    low_memory=True
)

features_reduced = umap_reducer.fit_transform(features_np)
features = torch.tensor(features_reduced, dtype=torch.float32)
del features_np, features_reduced
gc.collect()
torch.cuda.empty_cache() if torch.cuda.is_available() else None

num_nodes = features.shape[0]
chunk_size = 1000

min_non_zero = float('inf')
for i in range(0, num_nodes, chunk_size):
    chunk_end = min(i + chunk_size, num_nodes)
    chunk = features[i:chunk_end]

    dist_chunk = torch.cdist(chunk, features, p=2)

    zero_mask = (dist_chunk == 0)

    non_zero_distances = dist_chunk[~zero_mask]

    if non_zero_distances.numel() > 0:
        chunk_min = torch.min(non_zero_distances).item()
        if chunk_min < min_non_zero:
            min_non_zero = chunk_min

if min_non_zero == float('inf'):
    min_non_zero = 1.0

max_similarity = 1 / min_non_zero
special_value = max_similarity * 10

adj_sparse = lil_matrix((num_nodes, num_nodes), dtype=np.float32)

for i in range(0, num_nodes, chunk_size):
    chunk_end = min(i + chunk_size, num_nodes)
    chunk = features[i:chunk_end]

    dist_chunk = torch.cdist(chunk, features, p=2)

    zero_mask = (dist_chunk == 0)

    sim_chunk = torch.where(
        zero_mask,
        torch.full_like(dist_chunk, special_value),
        1 / dist_chunk
    )

    sim_chunk_np = sim_chunk.cpu().numpy()

    for j in range(sim_chunk_np.shape[0]):
        global_idx = i + j

        row_sims = sim_chunk_np[j]

        row_sims[global_idx] = 0

        topk_indices = np.argpartition(row_sims, -k)[-k:]

        for col_idx in topk_indices:
            if row_sims[col_idx] > 0:
                adj_sparse[global_idx, col_idx] = row_sims[col_idx]

    del dist_chunk, sim_chunk, sim_chunk_np
    gc.collect()


adj_sparse = adj_sparse.maximum(adj_sparse.T)

adj_normalized = normalize(adj_sparse.tocoo())

adj_coo = adj_normalized.tocoo()
rows = adj_coo.row
cols = adj_coo.col
values = adj_coo.data
indices = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
values = torch.tensor(values, dtype=torch.float)
shape = torch.Size(adj_coo.shape)
sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
output_file = f'D_adj_{name}.pt'
torch.save(sparse_tensor, output_file)