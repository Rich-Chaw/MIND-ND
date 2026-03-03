
'''
1. HM-GNN 架构实现方案

1.1 通道初始化与特征解耦（横向通道逻辑）

借鉴 PatchTST 的通道独立（Channel-independent）思想，我们不再将节点的所有特征（如度、介数、K-core、聚类系数等）直接拼接成一个长向量，而是将它们视为独立的信息通道。

输入分配：给定节点 $v$，其特征向量被拆分为 $K$ 个通道 $X^{(1)}, X^{(2)}, \dots, X^{(K)}$。例如，通道 1 仅包含度信息，通道 2 包含核心度（K-core）信息。

独立嵌入：每个通道通过一个专属的轻量级 MLP 将原始特征映射到统一的隐空间维度 $d$。这样做可以防止不同物理意义的拓扑特征在模型底层产生互干扰（即所谓的通道噪声）。

1.2 独立层次化路径（纵向尺度逻辑）

对于每一个通道 $k$，模型配备一个独立的层次化图编码器。这意味着每个通道都会经历自己的微观、中观和宏观抽象过程。

微观层（Micro-Scale）：在原图 $G$ 上使用该通道专属的 GNN（如轻量级 GAT）提取一阶邻域特征。

中观层（Meso-Scale）：利用通道相关的池化操作（如 AttPool）对节点进行聚类。

关键点：不同通道的池化逻辑是独立的。例如，"度通道"可能倾向于按中心性聚类，而"K-core通道"可能按结构核进行聚类。

池化后产生粗粒化图 $G'_{k}$，并再次应用 GNN 提取社区级模式。

宏观层（Macro-Scale）：通过全局 Readout 函数提取整图表示 $g^{(k)}$，捕捉该通道视角下的全局脆弱性。

1.3 跨通道关联增强与融合

在所有通道完成了各自的多尺度提取后，引入**跨通道注意力机制（Cross-Channel Attention, CCA）**来恢复被割裂的逻辑关联。

Query-Key 交互：使用一个通道（如中观尺度的社区桥接特征）作为 Query，去检索其他通道（如微观尺度的度特征）中的相关信息。

HSIC 约束：在训练中引入希尔伯特-施密特独立准则（HSIC）作为正则项，强制不同通道的层次化路径学习非冗余的特征，从而最大化特征的多样性。

'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm

from utils.graph_data import Batch
from .gnn_utils import _get_init_features, _read_out


# ---------- 通道独立嵌入：每个通道 (N,1) -> (N, hidden_dim) ----------
class ChannelEmbedding(nn.Module):
    """轻量级 MLP：将单通道原始特征映射到统一隐空间，避免通道间底层互干扰。"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, max(1, hidden_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, hidden_dim // 2), hidden_dim),
        )

    def forward(self, x):
        # x: (N, 1)
        return self.mlp(x)


# ---------- 单通道的层次化路径（简化版）：同一图上多层 GNN 拼接，无图粗化 ----------
class HierarchicalPathSimple(nn.Module):
    """
    对单一通道执行：多层 GNN 并拼接多尺度表征（每层在同一图上，节点数不变）。
    与 DiffPool 不同：无软聚类、无图粗化，仅为“多尺度特征拼接”的简化层次路径。
    """
    def __init__(self, hidden_dim, num_scales):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.embed = ChannelEmbedding(hidden_dim)
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(num_scales)
        ])
        # 多尺度拼接后投影到 (N, num_scales)，便于与其它通道对齐
        self.out_proj = nn.Linear(hidden_dim * num_scales, num_scales)

    def forward(self, x_k, edge_index, batch=None, num_nodes_b=None):
        h = self.embed(x_k) # (N, hidden_dim)
        scale_outputs = []
        for gnn in self.gnn_layers:
            h = gnn(h, edge_index)
            h = F.relu(h)
            scale_outputs.append(h)
        out = self.out_proj(torch.cat(scale_outputs, dim=-1))
        return out


# ---------- DiffPool 风格：学习软分配 S，图粗化 A'=S^T A S、X'=S^T H，在粗化图上做 GNN 再广播回节点 ----------
class HierarchicalPath(nn.Module):
    """
    DiffPool 风格层次路径：微观（原图 GNN）-> 中观（学习 S，粗化图 GNN，广播回节点）-> 宏观（粗图 readout 广播）。
    分配矩阵 S = softmax(AssignmentGNN(H))，图粗化 A'=S^T A S、X'=S^T H，在粗图上做 GNN 后用 S 广播回原节点数。
    """
    def __init__(self, hidden_dim, num_scales, n_clusters=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.n_clusters = n_clusters
        self.embed = ChannelEmbedding(hidden_dim)
        # 微观：原图上一阶 GNN
        self.gnn_micro = GCNConv(hidden_dim, hidden_dim)
        # 分配矩阵：由节点表示预测到 n_clusters 个簇的 logits，再按图做 softmax
        self.assignment_gnn = GCNConv(hidden_dim, n_clusters)
        # 中观：在粗化图上做 GNN（每图独立粗化，这里用共享的 GNN 处理粗图特征）
        self.gnn_meso = GCNConv(hidden_dim, hidden_dim)
        # 宏观：对粗图做 readout 后广播（无额外 GNN，用 meso 输出做 readout）
        # 输出：3 个尺度 [micro, meso_broadcast, macro_broadcast]，投影到 num_scales
        self.out_proj = nn.Linear(3 * hidden_dim, num_scales)

    def forward(self, x_k, edge_index, batch, num_nodes_b):
        """
        x_k: (total_nodes, 1), edge_index: (2, E), batch: (total_nodes,), num_nodes_b: (B,).
        仅对非 omni 节点做池化；omni 对应位置在 broadcast 时填 0 或图级 readout。
        """
        device = x_k.device
        dtype = x_k.dtype
        B = num_nodes_b.size(0)
        h = self.embed(x_k)  # (total_nodes, hidden_dim)

        # ----- 微观：整图（含 omni）上做一层 GNN -----
        h_micro = self.gnn_micro(h, edge_index)
        h_micro = F.relu(h_micro)  # (total_nodes, hidden_dim)

        # ----- 分配矩阵 S：由 h_micro 得到，按图做 softmax -----
        logits_S = self.assignment_gnn(h_micro, edge_index)  # (total_nodes, n_clusters)
        start_ids = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            num_nodes_b.cumsum(0)[:-1]
        ])
        # 每个图的非 omni 节点数 = num_nodes_b - 1
        n_non_omni = (num_nodes_b - 1).clamp(min=0)

        # ----- 中观 + 宏观：逐图做粗化、粗图 GNN、readout、广播回节点 -----
        h_meso_full = torch.zeros_like(h_micro)
        h_macro_full = torch.zeros_like(h_micro)
        for b in range(B):
            start = start_ids[b].item()
            n_b = n_non_omni[b].item()
            if n_b <= 0:
                continue
            # 当前图的非 omni 索引 [start, start+n_b)
            logits_b = logits_S[start : start + n_b]  # (n_b, n_clusters)
            S_b = F.softmax(logits_b, dim=-1)       # (n_b, n_c)
            H_b = h_micro[start : start + n_b]      # (n_b, F)
            # 子图邻接（仅非 omni 节点间边）
            mask = (edge_index[0] >= start) & (edge_index[0] < start + n_b) & \
                   (edge_index[1] >= start) & (edge_index[1] < start + n_b)
            local_ei = edge_index[:, mask] - start   # (2, E_b)
            n_c = min(self.n_clusters, n_b)
            S_b = S_b[:, :n_c]  # (n_b, n_c)
            A_b = torch.zeros(n_b, n_b, device=device, dtype=dtype)
            if local_ei.numel() > 0:
                A_b[local_ei[0], local_ei[1]] = 1.0
            A_prime = S_b.t() @ A_b @ S_b
            X_prime = S_b.t() @ H_b  # (n_c, F)
            # 粗图边（由 A_prime 得）
            ei_c = (A_prime > 1e-5).nonzero(as_tuple=False).t()
            if ei_c.numel() == 0:
                ei_c = torch.stack([
                    torch.arange(n_c, device=device),
                    torch.arange(n_c, device=device)
                ], dim=0)
            H_coarse = self.gnn_meso(X_prime, ei_c)
            H_coarse = F.relu(H_coarse)  # (n_c, F)
            h_meso_b = S_b @ H_coarse   # (n_b, F) 中观广播回节点
            h_meso_full[start : start + n_b] = h_meso_b
            # 宏观：粗图 readout（均值）再广播到该图所有非 omni 节点
            g_b = H_coarse.mean(dim=0)  # (F,)
            h_macro_full[start : start + n_b] = g_b.unsqueeze(0).expand(n_b, -1)

        # 三尺度拼接后投影到 (total_nodes, num_scales)
        scale_cat = torch.cat([h_micro, h_meso_full, h_macro_full], dim=-1)
        out = self.out_proj(scale_cat)
        return out


# ---------- 跨通道注意力 / 融合：堆叠的通道表征 (N, num_channels, num_scales) -> (N, num_channels*num_scales) ----------
class CrossChannelAttention_simple(nn.Module):
    """
    跨通道关联增强：对多通道堆叠特征做融合，恢复被通道解耦割裂的逻辑关联。
    简化实现：在通道维上做线性融合 + LayerNorm，输出与统一接口 (N, num_features*num_mps) 一致。
    """
    def __init__(self, num_channels, num_scales):
        super().__init__()
        self.num_channels = num_channels
        self.num_scales = num_scales
        dim = num_channels * num_scales
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
        )

    def forward(self, stacked_latents):
        """
        stacked_latents: (N, num_channels, num_scales)
        return: (N, num_channels * num_scales)
        """
        N = stacked_latents.size(0)
        x = stacked_latents.view(N, -1)
        return self.proj(x)


class CrossChannelAttention(nn.Module):
    """
    跨通道关联增强（Query-Key 注意力）：以通道为序列维度，做自注意力。
    Query-Key 交互：每个通道作为 Query 检索其他通道的 Key/Value，恢复被通道解耦割裂的逻辑关联。
    输入 (N, num_channels, num_scales)，输出 (N, num_channels*num_scales)。
    """
    def __init__(self, num_channels, num_scales, dropout=0.1):
        super().__init__()
        self.num_channels = num_channels
        self.num_scales = num_scales
        self.scale = num_scales ** -0.5
        # 每个通道的 Q、K、V 投影（对 num_scales 维做线性变换）
        self.proj_q = nn.Linear(num_scales, num_scales)
        self.proj_k = nn.Linear(num_scales, num_scales)
        self.proj_v = nn.Linear(num_scales, num_scales)
        self.proj_out = nn.Linear(num_scales, num_scales)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(num_channels * num_scales)

    def forward(self, stacked_latents):
        """
        stacked_latents: (N, num_channels, num_scales)
        return: (N, num_channels * num_scales)
        """
        N, C, S = stacked_latents.size()
        q = self.proj_q(stacked_latents)   # (N, C, S)
        k = self.proj_k(stacked_latents)
        v = self.proj_v(stacked_latents)
        # 通道维上的注意力：(N, C, S) @ (N, S, C) -> (N, C, C)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)   # (N, C, S)
        out = self.proj_out(out)      # (N, C, S)
        out = out.view(N, -1)
        return self.norm(out)


def hsic_linear(U, V):
    """
    线性核下的 HSIC 估计（Hilbert-Schmidt Independence Criterion）。
    用于约束不同通道表征尽量独立，最大化特征多样性。
    U: (N, d1), V: (N, d2)
    return: scalar, 越大表示越相关，训练时最小化该值。
    """
    N = U.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=U.device, dtype=U.dtype)
    H = torch.eye(N, device=U.device, dtype=U.dtype) - 1.0 / N
    K = U @ U.t()   # (N, N)
    L = V @ V.t()
    K_c = H @ K @ H
    L_c = H @ L @ H
    # HSIC = (1/N^2) * trace(K_c @ L_c)
    return (K_c * L_c).sum() / (N ** 2)


def hsic_regularizer(channel_latents):
    """
    多通道 HSIC 正则项：对通道两两计算 HSIC 并求和，鼓励通道间独立。
    channel_latents: list of (N, num_scales) 或 (N, C, S) 的 tensor。
    若为 (N, C, S)，先按通道拆成 list。
    return: scalar loss，训练时加上 lambda_hsic * 该值。
    """
    if isinstance(channel_latents, torch.Tensor):
        N, C, S = channel_latents.size()
        channel_latents = [channel_latents[:, c, :] for c in range(C)]
    total = torch.tensor(0.0, device=channel_latents[0].device, dtype=channel_latents[0].dtype)
    K = len(channel_latents)
    for i in range(K):
        for j in range(i + 1, K):
            total = total + hsic_linear(channel_latents[i], channel_latents[j])
    return total


class HM_GNN(nn.Module):
    """
    HM-GNN：PatchTST 式通道独立 + DiffPool 式层次化多尺度。
    - num_features 即 num_channels（每个特征维度视为一个通道）。
    - 每个通道独立经过 HierarchicalPath（Micro -> Meso -> Macro），再经 CCA 融合。
    - 与 MIND-ND 统一接口一致：forward(g: Batch) -> (N_non_omni, 2*num_features*num_mps)。
    """
    def __init__(self, num_features, num_heads, num_mps, positional_encoding=None, handcrafted_features=False, **kwargs):
        super().__init__()
        # num_features 作为通道数（PatchTST 中 channel-independent 的通道数）
        self.num_features = 5 if handcrafted_features else num_features
        self.num_heads = num_heads   # 用作每通道隐空间维度
        self.num_mps = num_mps      # 层次尺度数（Micro/Meso/Macro 对应 num_mps 层）
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features

        # 为每个通道定义专属的层次化路径（DiffPool 风格，可传 n_clusters 控制粗化簇数）
        n_clusters = kwargs.get("n_clusters", 8)
        self.paths = nn.ModuleList([
            HierarchicalPath(hidden_dim=num_heads, num_scales=num_mps, n_clusters=n_clusters)
            for _ in range(self.num_features)
        ])
        self.cca = CrossChannelAttention(num_channels=self.num_features, num_scales=num_mps)
        self.graph_norm = GraphNorm(self.num_features * num_mps, eps=1e-4)
        # HSIC 正则：训练时可用 get_hsic_regularizer() 取得，loss += lambda_hsic * model.get_hsic_regularizer()
        self.register_buffer("_hsic_loss", torch.tensor(0.0, dtype=torch.float32))

    def forward(self, g: Batch):
        """
        输入：Batch g（与其它 GNN 一致）。
        输出：与统一接口一致，即 _read_out(x_profile, g) -> (N, 2*num_features*num_mps)。
        """
        # 初始节点特征；(total_nodes, feat_dim)，feat_dim 为 num_features 或 5（handcrafted）
        x = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)

        # 1. 横向解耦：按通道拆成 X_list
        X_list = [x[:, k : k + 1] for k in range(self.num_features)]

        # 2. 纵向多尺度：每个通道独立走 HierarchicalPath
        channel_latents = []
        for k in range(self.num_features):
            g_k = self.paths[k](X_list[k], g.edge_index, g.batch, g.num_nodes_b)  # (total_nodes, num_mps)
            channel_latents.append(g_k)

        # 3. 堆叠为 (total_nodes, num_features, num_mps)
        stacked_latents = torch.stack(channel_latents, dim=1)

        # 4. HSIC 正则项（训练时最小化，促使各通道学习非冗余表征）
        self._hsic_loss.copy_(hsic_regularizer(stacked_latents))

        # 5. 跨通道关联增强（Query-Key 注意力）
        x_profile = self.cca(stacked_latents)  # (total_nodes, num_features*num_mps)
        x_profile = self.graph_norm(x_profile, g.batch)

        # 6. 与其它 GNN 一致：concat(node_emb, graph_emb)
        return _read_out(x_profile, g)

    def get_hsic_regularizer(self):
        """返回最近一次 forward 得到的 HSIC 正则标量，训练时：loss += lambda_hsic * model.get_hsic_regularizer()。"""
        return self._hsic_loss
