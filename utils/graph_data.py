import torch
import numpy as np

def ig_to_data(g):
    g = g.copy(); n = g.vcount()
    init_features = _extract_init_features(g, n)
    g.to_directed(); g.add_vertices(1); g.add_edges([(v_id, n) for v_id in range(n)])
    return Graph(np.array(g.get_edgelist(), dtype=np.int64).T, n, init_features=init_features)


def _extract_init_features(g, n):
    for attr_name in ("x_init", "init_features", "init_feature"):
        if attr_name in g.vs.attributes():
            values = g.vs[attr_name]
            if values is None or len(values) != n:
                continue
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(n, 1)
            if arr.ndim == 2 and arr.shape[0] == n:
                return arr
    return None


def _set_init_features(graphs, num_features, init_method='ONES', attr_name="x_init"):
    for g in graphs:
        if g is None:
            continue
        if g.vcount() == 0:
            g.vs[attr_name] = []
            continue
        if attr_name in g.vs.attributes():
            attr_vals = g.vs[attr_name]
            if attr_vals is not None and len(attr_vals) == g.vcount():
                continue
        n = g.vcount()
        if init_method == 'ONES':
            feats = np.ones((n, num_features), dtype=np.float32)
        elif init_method == 'RANDOM':
            feats = np.random.randn(n, num_features).astype(np.float32)
        else:
            raise ValueError(f"Unsupported init method: {init_method}")
        g.vs[attr_name] = [row.tolist() for row in feats]

class Graph:
    def __init__(self, edge_index, num_nodes, init_features=None):
        self.edge_index = edge_index   # == g.get_edgelist()^T  (2,g.ecount())
        self.num_nodes = num_nodes     # == g.vount()
        self.init_features = init_features

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

        self.x_init = None
        feat_dim = None
        for g in graph_array:
            if not hasattr(g, "init_features"):
                feat_dim = None
                break
            init_features = g.init_features
            if init_features is None:
                continue
            feats_np = np.asarray(init_features)
            feat_dim = int(feats_np.shape[1] if feats_np.ndim >= 2 else 1)
            break
        if feat_dim is not None:
            x_init = torch.zeros(self.total_nodes, feat_dim, dtype=torch.float32, device=self.device)
            for start_id, n_all, graph in zip(start_ids, num_nodes_b, graph_array):
                init_features = getattr(graph, "init_features", None)
                if init_features is None:
                    continue
                n_non_omni = int(n_all - 1)
                feats_np = np.asarray(init_features, dtype=np.float32)
                if feats_np.ndim == 1:
                    feats_np = feats_np.reshape(n_non_omni, 1)
                if feats_np.shape != (n_non_omni, feat_dim):
                    raise ValueError(
                        f"init_features shape mismatch: expected {(n_non_omni, feat_dim)}, got {feats_np.shape}"
                    )
                x_init[start_id:start_id + n_non_omni] = torch.as_tensor(
                    feats_np, dtype=torch.float32, device=self.device
                )
            self.x_init = x_init

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




