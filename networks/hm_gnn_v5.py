"""
HM-GNN v5: Simplified DiffPool-free hierarchical path with optional HSIC.

- 通道内路径：GCN 学习分配矩阵 S_b，通过 S_b 对节点特征做软聚类得到粗尺度表征，
  再广播回节点形成 (micro, meso, macro) 三尺度，无 dense_diff_pool、无稠密邻接矩阵。
- CCA 与 HSIC 正则与 v4 类似，HSIC 支持按图批量化，但在 v5 中默认关闭以节省算力。
- 接口保持一致：forward(g: Batch) -> (N_non_omni, 2*num_features*num_mps)。
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm

from utils.graph_data import Batch
from .gnn_utils import _get_init_features, _read_out


# ---------- 通道独立嵌入 ----------
class ChannelEmbedding(nn.Module):
    """单通道 (N, 1) -> (N, hidden_dim)。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, max(1, hidden_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, hidden_dim // 2), hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ---------- 通道内层次路径：GCN + 软聚类 S_b（无 DiffPool） ----------
class HierarchicalChannelPath(nn.Module):
    """
    单通道：微观 GCN -> 学习分配 S_b -> 软聚类得到粗尺度特征 -> 宏观 Readout，
    再广播回节点，得到 (N, num_scales)，不再使用 dense_diff_pool 和稠密邻接矩阵。
    """

    def __init__(self, hidden_dim: int, num_scales: int, num_clusters: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.num_clusters = num_clusters

        self.embed = ChannelEmbedding(hidden_dim)

        self.micro_gnn = GCNConv(hidden_dim, hidden_dim)
        self.pool_gnn = GCNConv(hidden_dim, num_clusters)
        # 在簇级别做非线性变换，代替在粗图上再跑一层 GCN
        self.coarse_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_proj = nn.Linear(hidden_dim * 3, num_scales)  # micro, meso, macro

    def forward(
        self,
        x_k: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        num_nodes_b: torch.Tensor,
    ):
        """
        x_k: (N, 1), edge_index: (2, E), batch: (N,), num_nodes_b: (B,).
        仅对非 omni 节点做聚类与广播；omni 位置在广播时填 0。
        """
        device = x_k.device
        dtype = x_k.dtype
        B = num_nodes_b.size(0)

        # 嵌入到通道隐藏空间
        h = self.embed(x_k)  # (N, hidden_dim)

        # 1. 微观：在原图上做一层 GCN
        h_micro = self.micro_gnn(h, edge_index)  # (N, hidden_dim)
        h_micro = F.relu(h_micro)

        # 2. 分配矩阵 S（按图 softmax），仅在非 omni 节点上使用
        logits_S = self.pool_gnn(h_micro, edge_index)  # (N, num_clusters)

        # 每个图的起始下标 & 非 omni 节点数
        start_ids = torch.cat(
            [
                torch.zeros(1, device=device, dtype=torch.long),
                num_nodes_b.cumsum(0)[:-1],
            ]
        )
        n_non_omni = (num_nodes_b - 1).clamp(min=0)

        h_meso_full = torch.zeros_like(h_micro)
        h_macro_full = torch.zeros_like(h_micro)

        # v5 不再构造稠密邻接矩阵，也不再调用 dense_diff_pool，
        # 仅在簇级别通过 S_b 做软聚类与广播。
        for b in range(B):
            start = start_ids[b].item()
            n_b = n_non_omni[b].item()
            if n_b <= 0:
                continue

            # 当前图的非 omni 节点范围
            node_slice = slice(start, start + n_b)

            # 当前图的分配 logits，裁剪到实际簇数 n_c
            logits_b = logits_S[node_slice]
            n_c = min(self.num_clusters, n_b)
            logits_b = logits_b[:, :n_c]
            S_b = F.softmax(logits_b, dim=-1)  # (n_b, n_c)

            # 当前图的微观特征
            H_b = h_micro[node_slice]  # (n_b, F)

            # 簇级特征：按列归一化的软聚类均值 H_coarse (n_c, F)
            # H_coarse[c] = sum_i S_b[i, c] * H_b[i] / sum_i S_b[i, c]
            cluster_mass = S_b.sum(dim=0, keepdim=True).clamp_min(1e-6)  # (1, n_c)
            H_coarse = (S_b.t() @ H_b) / cluster_mass.t()  # (n_c, F)

            # 在簇级别做非线性变换，模拟在粗图上跑一层 GNN
            H_coarse = self.coarse_mlp(H_coarse)  # (n_c, F)

            # 中观：将簇级特征通过 S_b 回投到节点级 (n_b, F)
            h_meso_b = S_b @ H_coarse  # (n_b, F)
            h_meso_full[node_slice] = h_meso_b

            # 宏观：对簇级特征做图级 readout，再广播回该图所有节点
            g_macro = H_coarse.mean(dim=0)  # (F,)
            h_macro_full[node_slice] = g_macro.unsqueeze(0).expand(n_b, -1)

        scale_cat = torch.cat([h_micro, h_meso_full, h_macro_full], dim=-1)
        out = self.out_proj(scale_cat)  # (N, num_scales)

        # v5 中不再使用 DiffPool 的 link/entropy 辅助损失，保持接口返回 0
        pool_loss = torch.tensor(0.0, device=device, dtype=dtype)
        return out, pool_loss


# ---------- 跨通道注意力 (CCA) ----------
class CrossChannelAttention(nn.Module):
    def __init__(self, num_channels: int, num_scales: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_channels = num_channels
        self.num_scales = num_scales
        # embed_dim 必须能被 num_heads 整除
        n_heads = min(num_heads, max(1, num_scales))
        while num_scales % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.attn = nn.MultiheadAttention(
            embed_dim=num_scales,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(num_channels * num_scales)
        self.proj = nn.Linear(num_channels * num_scales, num_channels * num_scales)

    def forward(self, stacked_latents: torch.Tensor) -> torch.Tensor:
        """
        stacked_latents: (N, num_channels, num_scales)
        return: (N, num_channels * num_scales)
        """
        N, C, S = stacked_latents.size()
        attn_out, _ = self.attn(stacked_latents, stacked_latents, stacked_latents)  # (N, C, S)
        out = attn_out.reshape(N, -1)
        out = self.norm(out)
        return self.proj(out)


# ---------- HSIC 正则项（按图批量化，可选使用） ----------
def _hsic_linear_single(U_b: torch.Tensor, V_b: torch.Tensor) -> torch.Tensor:
    """单图线性核 HSIC。U_b, V_b: (n_b, S)。"""
    n = U_b.size(0)
    if n <= 1:
        return torch.tensor(0.0, device=U_b.device, dtype=U_b.dtype)
    H = torch.eye(n, device=U_b.device, dtype=U_b.dtype) - 1.0 / n
    K = U_b @ U_b.t()
    L = V_b @ V_b.t()
    K_c = H @ K @ H
    L_c = H @ L @ H
    return (K_c * L_c).sum() / (n**2)


def hsic_linear_batched(
    U: torch.Tensor,
    V: torch.Tensor,
    start_ids: torch.Tensor,
    num_nodes_b: torch.Tensor,
) -> torch.Tensor:
    """
    按图批量化 HSIC：每张图内只在非 omni 节点 (n_b, S) 上计算 HSIC 再求和。
    U, V: (N, S); start_ids: (B,); num_nodes_b: (B,) 含 omni。每图非 omni 数 n_b = num_nodes_b - 1。
    """
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


def hsic_regularizer(
    channel_latents,
    start_ids: Optional[torch.Tensor] = None,
    num_nodes_b: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    多通道两两 HSIC 求和。
    channel_latents: list of (N, S) 或单个张量 (N, C, S)。
    若提供 start_ids, num_nodes_b（来自 Batch），则按图批量化计算，避免 N×N 显存。
    """
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


# ---------- HM-GNN v5 顶层：无 DiffPool、HSIC 可选（默认关闭） ----------
class HM_GNN_V5(nn.Module):
    """
    HM-GNN v5：GCN + 软聚类 S_b 层次路径，无 DiffPool；CCA 与 v4 类似，
    HSIC 正则支持按图批量化但默认关闭，以提高训练速度。
    forward(g: Batch) -> (N_non_omni, 2*num_features*num_mps)。
    """

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        num_mps: int,
        positional_encoding=None,
        handcrafted_features: bool = False,
        use_hsic: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_heads = num_heads
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features
        self.use_hsic = use_hsic

        n_clusters = kwargs.get("n_clusters", 8)

        self.paths = nn.ModuleList(
            [
                HierarchicalChannelPath(
                    hidden_dim=num_heads,
                    num_scales=num_mps,
                    num_clusters=n_clusters,
                )
                for _ in range(self.num_features)
            ]
        )
        self.cca = CrossChannelAttention(
            num_channels=self.num_features, num_scales=num_mps, num_heads=4
        )
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)
        self.register_buffer("_hsic_loss", torch.tensor(0.0, dtype=torch.float32))
        self._last_pool_loss = None  # 最近一次 forward 的 pool 损失

    def forward(self, g: Batch) -> torch.Tensor:
        x = _get_init_features(
            g,
            self.num_features,
            self.positional_encoding,
            self.handcrafted_features,
        )
        X_list = [x[:, k : k + 1] for k in range(self.num_features)]

        channel_latents = []
        pool_loss_acc = None
        for k in range(self.num_features):
            out_k, pool_loss_k = self.paths[k](X_list[k], g.edge_index, g.batch, g.num_nodes_b)
            channel_latents.append(out_k)
            pool_loss_acc = pool_loss_k if pool_loss_acc is None else (pool_loss_acc + pool_loss_k)
        self._last_pool_loss = pool_loss_acc

        stacked_latents = torch.stack(channel_latents, dim=1)  # (N, C, num_mps)

        # HSIC 默认关闭，仅在 use_hsic=True 时启用
        if self.use_hsic:
            self._hsic_loss.copy_(
                hsic_regularizer(
                    stacked_latents,
                    start_ids=g.start_ids,
                    num_nodes_b=g.num_nodes_b,
                ).detach()
            )
        else:
            # 保持形状与 dtype，清零数值
            self._hsic_loss.zero_()

        x_profile = self.cca(stacked_latents)  # (N, C*num_mps)
        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)

    def get_hsic_regularizer(self) -> torch.Tensor:
        return self._hsic_loss

    def get_pool_loss(self) -> torch.Tensor:
        """返回最近一次 forward 的池化辅助损失（v5 中恒为 0，但保留接口便于统一调用）。"""
        if self._last_pool_loss is not None:
            return self._last_pool_loss
        return torch.tensor(0.0, device=next(self.parameters()).device)

"""
HM-GNN v3: Batched variant of v2 to avoid CUDA OOM.

- Same structure as v2: GAT + DiffPool + CCA + HSIC.
- HSIC 按图批量化：在每张图内 (n_b, S) 上计算 HSIC，再求和，避免整批 N×N 矩阵 (OOM)。
- 使用 utils/graph_data.Batch 的 start_ids / num_nodes_b 做按图切片，与 graph_data 批数据思路一致。
- 接口与 v2 一致：forward(g: Batch) -> (N_non_omni, 2*num_features*num_mps)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm
from torch_geometric.nn.dense import dense_diff_pool

from utils.graph_data import Batch
from .gnn_utils import _get_init_features, _read_out


# ---------- 通道独立嵌入 ----------
class ChannelEmbedding(nn.Module):
    """单通道 (N, 1) -> (N, hidden_dim)，避免通道间底层互干扰。"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, max(1, hidden_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, hidden_dim // 2), hidden_dim),
        )

    def forward(self, x):
        return self.mlp(x)


# ---------- 通道内层次路径：GCN + DiffPool，输出节点级多尺度表征（批量化） ----------
class HierarchicalChannelPath(nn.Module):
    """
    单通道：微观 GAT -> 学习分配 S -> DiffPool 粗化 -> 中观 GAT -> 宏观 Readout，
    再广播回节点，得到 (N, num_scales) 与 pool_loss。
    """
    def __init__(self, hidden_dim, num_scales, num_clusters=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.num_clusters = num_clusters
        self.embed = ChannelEmbedding(hidden_dim)

        self.micro_gnn = GCNConv(hidden_dim, hidden_dim)
        self.pool_gnn = GCNConv(hidden_dim, num_clusters)
        self.meso_gnn = GCNConv(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim * 3, num_scales)  # micro, meso, macro

    def forward(self, x_k, edge_index, batch, num_nodes_b):
        """
        x_k: (N, 1), edge_index: (2, E), batch: (N,), num_nodes_b: (B,).
        仅对非 omni 节点做池化；omni 位置在广播时填 0。
        """
        device = x_k.device
        dtype = x_k.dtype
        B = num_nodes_b.size(0)
        h = self.embed(x_k)  # (N, hidden_dim)

        # 1. 微观
        h_micro = self.micro_gnn(h, edge_index) # (N, hidden_dim)
        h_micro = F.relu(h_micro)

        # 2. 分配矩阵 S（按图 softmax）
        logits_S = self.pool_gnn(h_micro, edge_index)  # (N, num_clusters)
        start_ids = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            num_nodes_b.cumsum(0)[:-1]
        ])
        n_non_omni = (num_nodes_b - 1).clamp(min=0)

        h_meso_full = torch.zeros_like(h_micro)
        h_macro_full = torch.zeros_like(h_micro)
        total_link_loss = torch.tensor(0.0, device=device, dtype=dtype)
        total_ent_loss = torch.tensor(0.0, device=device, dtype=dtype)

        for b in range(B):
            start = start_ids[b].item()
            n_b = n_non_omni[b].item()
            if n_b <= 0:
                continue

            logits_b = logits_S[start : start + n_b]
            n_c = min(self.num_clusters, n_b)
            logits_b = logits_b[:, :n_c]
            S_b = F.softmax(logits_b, dim=-1)  # (n_b, n_c)
            H_b = h_micro[start : start + n_b]  # (n_b, F)

            mask_b = (edge_index[0] >= start) & (edge_index[0] < start + n_b) & \
                     (edge_index[1] >= start) & (edge_index[1] < start + n_b)
            local_ei = edge_index[:, mask_b] - start
            A_b = torch.zeros(n_b, n_b, device=device, dtype=dtype)
            if local_ei.numel() > 0:
                A_b[local_ei[0], local_ei[1]] = 1.0

            # DiffPool 需要 (B, N, F), (B, N, N), (B, N, C), mask
            x_dense = H_b.unsqueeze(0)   # (1, n_b, F)
            adj_dense = A_b.unsqueeze(0) # (1, n_b, n_b)
            s_dense = S_b.unsqueeze(0)   # (1, n_b, n_c)
            mask_dense = torch.ones(1, n_b, dtype=torch.bool, device=device)

            x_pooled, adj_pooled, l_link, l_ent = dense_diff_pool(
                x_dense, adj_dense, s_dense, mask_dense
            )
            total_link_loss = total_link_loss + l_link.sum()
            total_ent_loss = total_ent_loss + l_ent.sum()

            # 中观：在粗图上做 GAT
            x_c = x_pooled[0]   # (n_c, F)
            adj_c = adj_pooled[0]
            ei_c = (adj_c > 1e-5).nonzero(as_tuple=False).t()
            if ei_c.numel() == 0:
                ei_c = torch.stack([
                    torch.arange(n_c, device=device),
                    torch.arange(n_c, device=device)
                ], dim=0)
            H_coarse = self.meso_gnn(x_c, ei_c)
            H_coarse = F.relu(H_coarse)  # (n_c, F)
            h_meso_b = S_b @ H_coarse   # (n_b, F)
            h_meso_full[start : start + n_b] = h_meso_b

            # 宏观：粗图 readout 再广播
            g_macro = H_coarse.mean(dim=0)
            h_macro_full[start : start + n_b] = g_macro.unsqueeze(0).expand(n_b, -1)

        scale_cat = torch.cat([h_micro, h_meso_full, h_macro_full], dim=-1)
        out = self.out_proj(scale_cat)  # (N, num_scales)
        pool_loss = total_link_loss + total_ent_loss
        return out, pool_loss


# ---------- 跨通道注意力 (CCA)：MultiheadAttention 融合 ----------
class CrossChannelAttention(nn.Module):
    """
    对 (N, num_channels, num_scales) 在通道维做 Self-Attention (CCA)，
    输出 (N, num_channels * num_scales) 与统一接口对齐。
    """
    def __init__(self, num_channels, num_scales, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.num_scales = num_scales
        # embed_dim 必须能被 num_heads 整除
        n_heads = min(num_heads, max(1, num_scales))
        while num_scales % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.attn = nn.MultiheadAttention(
            embed_dim=num_scales, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(num_channels * num_scales)
        self.proj = nn.Linear(num_channels * num_scales, num_channels * num_scales)

    def forward(self, stacked_latents):
        """
        stacked_latents: (N, num_channels, num_scales)
        return: (N, num_channels * num_scales)
        """
        N, C, S = stacked_latents.size()
        attn_out, _ = self.attn(stacked_latents, stacked_latents, stacked_latents)  # (N, C, S)
        out = attn_out.reshape(N, -1)
        out = self.norm(out)
        return self.proj(out)


# ---------- HSIC 正则项（v3：按图批量化，避免 N×N OOM）----------
def _hsic_linear_single(U_b, V_b):
    """单图线性核 HSIC。U_b, V_b: (n_b, S)，仅用于小 n_b，避免大矩阵。"""
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
    """
    按图批量化 HSIC：每张图内只在非 omni 节点 (n_b, S) 上计算 HSIC 再求和。
    U, V: (N, S); start_ids: (B,); num_nodes_b: (B,) 含 omni。每图非 omni 数 n_b = num_nodes_b - 1。
    """
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
    """
    多通道两两 HSIC 求和。
    channel_latents: list of (N, S) 或 (N, C, S)。
    若提供 start_ids, num_nodes_b（来自 Batch），则按图批量化计算，避免 N×N 显存。
    """
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


# ---------- HM-GNN v3 顶层：批量化 HSIC，与 MIND-ND 统一接口 ----------
class HM_GNN_V3(nn.Module):
    """
    HM-GNN v3：与 v2 相同结构，HSIC 按图批量化以节省显存。
    forward(g: Batch) -> (N_non_omni, 2*num_features*num_mps)。
    """
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False, **kwargs):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_heads = num_heads
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features
        n_clusters = kwargs.get("n_clusters", 8)

        self.paths = nn.ModuleList([
            HierarchicalChannelPath(
                hidden_dim=num_heads, num_scales=num_mps, num_clusters=n_clusters
            )
            for _ in range(self.num_features)
        ])
        self.cca = CrossChannelAttention(
            num_channels=self.num_features, num_scales=num_mps, num_heads=4
        )
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)
        self.register_buffer("_hsic_loss", torch.tensor(0.0, dtype=torch.float32))
        self._last_pool_loss = None  # 最近一次 forward 的 pool 损失，保留计算图供训练

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

        stacked_latents = torch.stack(channel_latents, dim=1)  # (N, C, num_mps)
        self._hsic_loss.copy_(hsic_regularizer(
            stacked_latents,
            start_ids=g.start_ids,
            num_nodes_b=g.num_nodes_b,
        ).detach())

        x_profile = self.cca(stacked_latents)  # (N, C*num_mps)
        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)

    def get_hsic_regularizer(self):
        return self._hsic_loss

    def get_pool_loss(self):
        """返回最近一次 forward 的 DiffPool 辅助损失（link + entropy），可参与反传。"""
        return self._last_pool_loss if self._last_pool_loss is not None else torch.tensor(0.0, device=next(self.parameters()).device)
