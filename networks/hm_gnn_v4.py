"""
HM-GNN v4: GCN + ASAPool for fast batched pooling (no per-graph loop).

- HierarchicalChannelPath 使用 GCN + ASAPool：稀疏池化、整批一次前向，避免 v3 的
  逐图 DiffPool + 稠密矩阵导致的极慢速度。
- CCA + 按图批量化 HSIC 与 v3 相同。
- 接口一致：forward(g: Batch) -> (N_non_omni, 2*num_features*num_mps)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm
from torch_geometric.nn.pool import ASAPooling
from torch_scatter import scatter_mean

from utils.graph_data import Batch
from .gnn_utils import _get_init_features, _read_out


# ---------- 通道独立嵌入 ----------
class ChannelEmbedding(nn.Module):
    """单通道 (N, 1) -> (N, hidden_dim)。"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, max(1, hidden_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, hidden_dim // 2), hidden_dim),
        )

    def forward(self, x):
        return self.mlp(x)


# ---------- 通道内层次路径：GCN + ASAPool（整批稀疏池化，无逐图循环）----------
class HierarchicalChannelPath(nn.Module):
    """
    单通道：微观 GCN -> ASAPool 粗化 -> 中观 GCN -> 宏观 readout，
    再按 perm 与 batch 广播回所有节点，得到 (N, num_scales)。无 DiffPool 辅助损失。
    """
    def __init__(self, hidden_dim, num_scales, ratio=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.embed = ChannelEmbedding(hidden_dim)

        self.micro_gnn = GCNConv(hidden_dim, hidden_dim)
        self.asapool = ASAPooling(
            in_channels=hidden_dim,
            ratio=ratio,
            GNN=GCNConv,
        )
        self.meso_gnn = GCNConv(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * 3, num_scales)  # micro, meso, macro

    def forward(self, x_k, edge_index, batch, num_nodes_b):
        """
        x_k: (N, 1), edge_index: (2, E), batch: (N,), num_nodes_b: (B,).
        整批一次 ASAPool，无 per-graph 循环。
        """
        N = x_k.size(0)
        device = x_k.device
        dtype = x_k.dtype
        B = num_nodes_b.size(0)

        h = self.embed(x_k)  # (N, hidden_dim)

        # 1. 微观
        h_micro = self.micro_gnn(h, edge_index)
        h_micro = F.relu(h_micro)

        # 2. ASAPool：稀疏池化，整批一次
        x_pooled, edge_pooled, _, batch_pooled, perm = self.asapool(
            h_micro, edge_index, batch=batch
        )
        # x_pooled (num_kept, F), perm (num_kept,) 为保留的节点在原图中的下标

        num_kept = x_pooled.size(0)
        h_meso_full = torch.zeros_like(h_micro)
        h_macro_full = torch.zeros_like(h_micro)

        if num_kept > 0:
            # 3. 中观：在粗图上做 GCN
            H_coarse = self.meso_gnn(x_pooled, edge_pooled)
            H_coarse = F.relu(H_coarse)  # (num_kept, F)

            # 每图粗节点特征的均值，再广播回该图所有节点
            graph_mean = scatter_mean(H_coarse, batch_pooled, dim=0, dim_size=B)  # (B, F)
            h_meso_full = graph_mean[batch].clone()
            h_meso_full[perm] = H_coarse
            h_macro_full = graph_mean[batch]

        scale_cat = torch.cat([h_micro, h_meso_full, h_macro_full], dim=-1)
        out = self.out_proj(scale_cat)  # (N, num_scales)
        pool_loss = torch.tensor(0.0, device=device, dtype=dtype)
        return out, pool_loss


# ---------- 跨通道注意力 (CCA) ----------
class CrossChannelAttention(nn.Module):
    def __init__(self, num_channels, num_scales, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.num_scales = num_scales
        n_heads = min(num_heads, max(1, num_scales))
        while num_scales % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.attn = nn.MultiheadAttention(
            embed_dim=num_scales, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(num_channels * num_scales)
        self.proj = nn.Linear(num_channels * num_scales, num_channels * num_scales)

    def forward(self, stacked_latents):
        N, C, S = stacked_latents.size()
        attn_out, _ = self.attn(stacked_latents, stacked_latents, stacked_latents)
        out = attn_out.reshape(N, -1)
        out = self.norm(out)
        return self.proj(out)


# ---------- HSIC 正则项（按图批量化）----------
def _hsic_linear_single(U_b, V_b):
    n = U_b.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=U_b.device, dtype=U_b.dtype)
    H = torch.eye(n, device=U_b.device, dtype=U_b.dtype) - 1.0 / n
    K = U_b @ U_b.t()
    L = V_b @ V_b.t()
    K_c = H @ K @ H
    L_c = H @ L @ H
    return (K_c * L_c).sum() / (n ** 2)


def hsic_linear_batched(U, V, start_ids, num_nodes_b):
    device, dtype = U.device, U.dtype
    B = num_nodes_b.size(0)
    n_non_omni = (num_nodes_b - 1).clamp(min=0)
    total = torch.tensor(0.0, device=device, dtype=dtype)
    for b in range(B):
        n_b = n_non_omni[b].item()
        if n_b <= 1:
            continue
        start = start_ids[b].item()
        U_b = U[start : start + n_b]
        V_b = V[start : start + n_b]
        total = total + _hsic_linear_single(U_b, V_b)
    return total


def hsic_regularizer(channel_latents, start_ids=None, num_nodes_b=None):
    if isinstance(channel_latents, torch.Tensor):
        N, C, S = channel_latents.size()
        channel_latents = [channel_latents[:, c, :] for c in range(C)]
    total = torch.tensor(0.0, device=channel_latents[0].device, dtype=channel_latents[0].dtype)
    K = len(channel_latents)
    batched = start_ids is not None and num_nodes_b is not None
    for i in range(K):
        for j in range(i + 1, K):
            if batched:
                total = total + hsic_linear_batched(
                    channel_latents[i], channel_latents[j], start_ids, num_nodes_b
                )
            else:
                total = total + _hsic_linear_single(channel_latents[i], channel_latents[j])
    return total


# ---------- HM-GNN v4 顶层 ----------
class HM_GNN_V4(nn.Module):
    """
    HM-GNN v4：GCN + ASAPool 层次路径，整批稀疏池化，无逐图循环，训练更快。
    forward(g: Batch) -> (N_non_omni, 2*num_features*num_mps)。
    """
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False, **kwargs):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_heads = num_heads
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features
        ratio = kwargs.get("pool_ratio", 0.5)

        self.paths = nn.ModuleList([
            HierarchicalChannelPath(
                hidden_dim=num_heads, num_scales=num_mps, ratio=ratio
            )
            for _ in range(self.num_features)
        ])
        self.cca = CrossChannelAttention(
            num_channels=self.num_features, num_scales=num_mps, num_heads=4
        )
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)
        self.register_buffer("_hsic_loss", torch.tensor(0.0, dtype=torch.float32))
        self._last_pool_loss = None

    def forward(self, g: Batch):
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        X_list = [x[:, k : k + 1] for k in range(self.num_features)]

        channel_latents = []
        pool_loss_acc = None
        for k in range(self.num_features):
            out_k, pool_loss_k = self.paths[k](X_list[k], g.edge_index, g.batch, g.num_nodes_b)
            channel_latents.append(out_k)
            pool_loss_acc = pool_loss_k if pool_loss_acc is None else (pool_loss_acc + pool_loss_k)
        self._last_pool_loss = pool_loss_acc

        stacked_latents = torch.stack(channel_latents, dim=1)
        self._hsic_loss.copy_(hsic_regularizer(
            stacked_latents,
            start_ids=g.start_ids,
            num_nodes_b=g.num_nodes_b,
        ).detach())

        x_profile = self.cca(stacked_latents)
        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)

    def get_hsic_regularizer(self):
        return self._hsic_loss

    def get_pool_loss(self):
        return self._last_pool_loss if self._last_pool_loss is not None else torch.tensor(0.0, device=next(self.parameters()).device)
