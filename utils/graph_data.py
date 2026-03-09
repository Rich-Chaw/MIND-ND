import torch
import numpy as np

def ig_to_data(g):
    g = g.copy(); n = g.vcount()
    g.to_directed(); g.add_vertices(1); g.add_edges([(v_id, n) for v_id in range(n)])
    return Graph(np.array(g.get_edgelist(), dtype=np.int64).T, n)

class Graph:
    def __init__(self, edge_index, num_nodes):
        self.edge_index = edge_index   # == g.get_edgelist()^T  (2,g.ecount())
        self.num_nodes = num_nodes     # == g.vount()

# a Batch graphs as a Batch
class Batch:
    def __init__(self, device, graph_array):
        '''
        graph_array List[Graph]
        N: sum of nodes num in graph array N=N_1 + N_2 + N_3 ....., Note N_i is the node num before add omni node
        E: sum of edges num in graph array E=E_1 + E_2 + E_3 ....., Note E_i is the edge num before add omni node
        '''
        self.device = device
        self.batch_size = len(graph_array) # B

        # each graph will add an omni-node 
        num_nodes_b = np.array([g.num_nodes+1 for g in graph_array], dtype=np.int64)  #  (B)  data:[N_1+1, N_2+1, ...]
        #  (B)
        start_ids = np.zeros(self.batch_size, dtype=np.int64)   #(B)  data:[0, N_1+1, N_1+N_2+2, ...]
        start_ids[1:] = np.cumsum(num_nodes_b[:-1])

        act_offsets = start_ids - np.arange(self.batch_size)

        omni_ids = start_ids + num_nodes_b - 1                  #(B)  data:[N_1, N_1+N_2+1, N_1+N_2+N_3+2, ...]

        batch = np.repeat(np.arange(self.batch_size), num_nodes_b, axis=0) #(N+B)     data:[0,0,..,1,1,,...,B-1,B-1,...]

        edge_index = np.concatenate([g.edge_index+s for s, g in zip(start_ids, graph_array)], axis=1)   #(2,E+N1+N2+..)

        self.num_nodes_b = torch.tensor(num_nodes_b, dtype=torch.long, device=self.device)
        self.total_nodes = self.num_nodes_b.sum() # N+B
        self.start_ids = torch.tensor(start_ids, dtype=torch.long, device=self.device)  # (B), 供 GNN 批量化复用

        self.act_offsets = torch.tensor(act_offsets, dtype=torch.long, device=self.device) # (B)

        self.omni_ids = torch.tensor(omni_ids, dtype=torch.long, device=self.device)  # (B)
        self.non_omni_mask = torch.ones(self.total_nodes, dtype=torch.bool, device=self.device) #(N+B)
        self.non_omni_mask[self.omni_ids] = False

        self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        self.batch = torch.tensor(batch, dtype=torch.long, device=self.device) #(N+B)
        self.batch_non_omni = self.batch[self.non_omni_mask] #(N)


def random_walk_positional_encoding(batch, num_features, device):
    """
    Compute Random Walk Positional Encodings for a batch of graphs.
    For each node i, the feature is the probability that a random walk starting at i
    returns to i after k steps, for k=1, 2, ..., num_features.

    Returns:
        Tensor of shape (N, F) on the given device.
        Omni-nodes get zero encoding.
    """
    B = batch.batch_size
    num_nodes_b = batch.num_nodes_b  # (B,) includes omni
    omni_ids = batch.omni_ids  # (B,)
    edge_index = batch.edge_index  # (2, E)
    # start_ids[b] = index of first node of graph b
    start_ids = torch.zeros(B, dtype=torch.long, device=device)
    if B > 1:
        start_ids[1:] = torch.cumsum(num_nodes_b[:-1], dim=0)

    out_list = []
    for b in range(B):
        start_id = start_ids[b].item()
        n_b = (num_nodes_b[b] - 1).item()  # non-omni nodes
        omni_id_b = omni_ids[b].item()
        if n_b == 0:
            # graph has 0 original nodes, only omni
            pe = torch.zeros(1, num_features, device=device)
            out_list.append(pe)
            continue
        # Edges entirely within non-omni nodes (exclude omni)
        mask = (
            (edge_index[0] >= start_id) & (edge_index[0] < start_id + n_b) &
            (edge_index[1] >= start_id) & (edge_index[1] < start_id + n_b)
        )
        local_ei = edge_index[:, mask] - start_id  # (2, E_b)
        if local_ei.shape[1] == 0:
            # no edges: P^k = 0 for k>=1, use zeros
            pe = torch.zeros(n_b + 1, num_features, device=device)
            out_list.append(pe)
            continue
        # Build adjacency A (n_b x n_b)
        A = torch.zeros(n_b, n_b, device=device)
        A[local_ei[0], local_ei[1]] = 1.0
        deg = A.sum(dim=1, keepdim=True).clamp(min=1e-8)
        P = A / deg  # transition matrix
        # Diagonals of P^1, P^2, ..., P^num_features
        diags = [torch.diag(P)]
        p_power = P.clone()
        for _ in range(2, num_features + 1):
            p_power = p_power @ P
            diags.append(torch.diag(p_power))
        pe_non_omni = torch.stack(diags, dim=1)  # (n_b, num_features)
        omni_row = torch.zeros(1, num_features, device=device)
        pe = torch.cat([pe_non_omni, omni_row], dim=0)  # (n_b+1, num_features)
        out_list.append(pe)
    return torch.cat(out_list, dim=0)


def handcrafted_node_features(batch, device):
    """
    Compute handcrafted node features for a batch of graphs.
    Features (5 dims): degree, average degree of neighbor, local clustering, k-core, 1 (normalization).
    Omni-nodes get zero features.
    Batched: degree and avg_deg_neighbor in one shot; clustering and k-core per graph with vectorized torch.

    Returns:
        Tensor of shape (N, 5) on the given device.
    """
    B = batch.batch_size
    num_nodes_b = batch.num_nodes_b.cpu().numpy()  # (B,) includes omni
    edge_index_np = batch.edge_index.cpu().numpy()  # (2, E)
    start_ids = np.zeros(B, dtype=np.int64)
    if B > 1:
        start_ids[1:] = np.cumsum(num_nodes_b[:-1])

    out_list = []
    for b in range(B):
        start_id = int(start_ids[b])
        n_b = int(num_nodes_b[b] - 1)  # non-omni nodes
        if n_b == 0:
            feats = np.zeros((1, 5), dtype=np.float32)
            out_list.append(feats)
            continue
        mask = (
            (edge_index_np[0] >= start_id) & (edge_index_np[0] < start_id + n_b) &
            (edge_index_np[1] >= start_id) & (edge_index_np[1] < start_id + n_b)
        )
        local_ei = edge_index_np[:, mask] - start_id  # (2, E_b)
        feats_b = _handcrafted_single_graph(local_ei, n_b)
        omni_row = np.zeros((1, 5), dtype=np.float32)
        feats = np.concatenate([feats_b, omni_row], axis=0)
        out_list.append(feats)
    out = np.concatenate(out_list, axis=0)
    return torch.tensor(out, dtype=torch.float32, device=device)


def _handcrafted_single_graph(edge_index, n):
    """Compute 5 handcrafted features for one graph: degree, avg_deg_neighbor, clustering, k_core, 1."""
    src, dst = edge_index[0], edge_index[1]
    deg = np.bincount(src, minlength=n).astype(np.float64)
    adj_list = [[] for _ in range(n)]
    for i in range(edge_index.shape[1]):
        u, v = int(src[i]), int(dst[i])
        adj_list[u].append(v)

    avg_deg_neighbor = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if deg[i] > 0:
            nb = np.array(adj_list[i], dtype=np.int64)
            avg_deg_neighbor[i] = np.mean(deg[nb])

    triangle_count = np.zeros(n, dtype=np.float64)
    for i in range(edge_index.shape[1]):
        u, v = int(src[i]), int(dst[i])
        if u >= v:
            continue
        set_u = set(adj_list[u])
        set_v = set(adj_list[v])
        for c in set_u & set_v:
            triangle_count[c] += 1
    clustering = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if deg[i] >= 2:
            clustering[i] = triangle_count[i] / (deg[i] * (deg[i] - 1) / 2.0)

    deg_int = np.maximum(deg.astype(np.int64), 0)
    deg_current = deg_int.copy()
    core = np.zeros(n, dtype=np.float64)
    remaining = np.ones(n, dtype=bool)
    while np.any(remaining):
        idx_rem = np.where(remaining)[0]
        deg_rem = deg_current[remaining]
        i = idx_rem[np.argmin(deg_rem)]
        k_val = float(deg_current[i])
        core[i] = k_val
        remaining[i] = False
        for j in adj_list[i]:
            if remaining[j] and deg_current[j] > 0:
                deg_current[j] -= 1

    ones = np.ones(n, dtype=np.float64)
    return np.stack([deg, avg_deg_neighbor, clustering, core, ones], axis=1).astype(np.float32)





# def handcrafted_node_features(batch, device):
#     """
#     Compute handcrafted node features for a batch of graphs.
#     Features (5 dims): degree, average degree of neighbor, local clustering, k-core, 1 (normalization).
#     Omni-nodes get zero features.
#     Batched: degree and avg_deg_neighbor in one shot; clustering and k-core per graph with vectorized torch.

#     Returns:
#         Tensor of shape (N, 5) on the given device.
#     """
#     B = batch.batch_size
#     edge_index = batch.edge_index  # (2, E)
#     non_omni = batch.non_omni_mask  # (total_nodes,)
#     start_ids = batch.start_ids  # (B,)
#     num_nodes_b = batch.num_nodes_b  # (B,) includes omni
#     total_nodes = batch.total_nodes

#     # Only edges between non-omni nodes (subgraph for degree/clustering)
#     mask = non_omni[edge_index[0]] & non_omni[edge_index[1]]
#     ei = edge_index[:, mask]  # (2, E')
#     src, dst = ei[0], ei[1]

#     # --- Batch-level: degree and avg_deg_neighbor in one shot (no per-graph loop) ---
#     deg = torch.zeros(total_nodes, dtype=torch.float32, device=device)
#     deg.scatter_add_(0, src, torch.ones(ei.size(1), dtype=torch.float32, device=device))
#     sum_deg_neighbor = torch.zeros(total_nodes, dtype=torch.float32, device=device)
#     sum_deg_neighbor.scatter_add_(0, src, deg[dst])
#     deg_safe = deg.clamp(min=1e-8)
#     avg_deg_neighbor = sum_deg_neighbor / deg_safe

#     # Output tensor: (total_nodes, 5)
#     feats = torch.zeros(total_nodes, 5, dtype=torch.float32, device=device)
#     feats[:, 0] = deg
#     feats[:, 1] = avg_deg_neighbor
#     feats[non_omni, 4] = 1.0

#     # --- Per-graph: clustering and k-core (vectorized torch, no Python inner loops) ---
#     for b in range(B):
#         start_id = start_ids[b].item()
#         n_b = (num_nodes_b[b] - 1).item()
#         if n_b <= 0:
#             continue
#         # Local edge index in [0, n_b)
#         m = (ei[0] >= start_id) & (ei[0] < start_id + n_b) & (ei[1] >= start_id) & (ei[1] < start_id + n_b)
#         local_ei = ei[:, m] - start_id  # (2, E_b)
#         clustering_b, core_b = _handcrafted_clustering_kcore_torch(local_ei, n_b, device)
#         feats[start_id : start_id + n_b, 2] = clustering_b
#         feats[start_id : start_id + n_b, 3] = core_b

#     # Omni-nodes: zero (feats already zeros; deg/avg_deg are 0 for omni since they're not in ei)
#     return feats


# def _handcrafted_clustering_kcore_torch(edge_index, n, device):
#     """
#     Vectorized clustering and k-core for one graph. edge_index: (2, E) local indices in [0, n).
#     Returns clustering (n,), core (n,) on device.
#     Clustering: sparse matmul O(E*n); k-core: packed adjacency O(deg) per iteration.
#     """
#     src, dst = edge_index[0], edge_index[1]
#     E = edge_index.size(1)
#     deg = torch.zeros(n, dtype=torch.float32, device=device)
#     deg.scatter_add_(0, src, torch.ones(E, dtype=torch.float32, device=device))
#     deg_int = deg.long().clamp(min=0)

#     # Clustering: A2 = A @ A via sparse @ dense to avoid O(n^3). Skip if n too large to avoid O(n^2) mem.
#     if E == 0:
#         triangle_count = torch.zeros(n, dtype=torch.float32, device=device)
#     else:
#         A_sparse = torch.sparse_coo_tensor(
#             torch.stack([src, dst]),
#             torch.ones(E, dtype=torch.float32, device=device),
#             (n, n),
#         )
#         A_dense = torch.zeros(n, n, dtype=torch.float32, device=device)
#         A_dense[src, dst] = 1.0
#         A2 = torch.sparse.mm(A_sparse, A_dense)  # O(E*n) instead of O(n^3)
#         triangle_count = (A2 * A_dense).sum(dim=1) * 0.5  # (A2*A).sum = 2*num_triangles
#     denom = deg * (deg - 1).clamp(min=1e-8)
#     clustering = (2.0 * triangle_count / denom).clamp(max=1.0)

#     # K-core: packed adjacency so neighbors of i are dst_sorted[offset[i]:offset[i+1]]
#     perm = src.argsort()
#     dst_sorted = dst[perm]
#     offset = torch.cat([
#         torch.zeros(1, dtype=torch.long, device=device),
#         deg_int.cumsum(0),
#     ])

#     deg_current = deg_int.clone()
#     core = torch.zeros(n, dtype=torch.float32, device=device)
#     remaining = torch.ones(n, dtype=torch.bool, device=device)
#     inf_f32 = torch.tensor(float('inf'), dtype=torch.float32, device=device)
#     for _ in range(n):
#         if not remaining.any():
#             break
#         deg_rem = torch.where(remaining, deg_current.float(), inf_f32)
#         i = deg_rem.argmin().item()
#         if not remaining[i]:
#             break
#         core[i] = float(deg_current[i])
#         remaining[i] = False
#         lo, hi = offset[i].item(), offset[i + 1].item()
#         if lo < hi:
#             j_nodes = dst_sorted[lo:hi]
#             j_remaining = j_nodes[remaining[j_nodes]]
#             if j_remaining.numel() > 0:
#                 deg_current[j_remaining] = (deg_current[j_remaining] - 1).clamp(min=0)

#     return clustering, core