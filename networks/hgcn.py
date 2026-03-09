"""
Hyperbolic GCN (HGCN) on Poincaré ball，参考 hgcn/models/encoders.py 与 layers/hyp_layers。
支持 edge_index 输入，用于图 VAE 编码器与双曲距离解码。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


# ---------- 双曲数学 (参考 hgcn/utils/math_utils) ----------
def _artanh(x):
    x = x.clamp(-1 + 1e-15, 1 - 1e-15)
    return (torch.log1p(x) - torch.log1p(-x)).mul(0.5)


def _arcosh(x):
    x = x.clamp(min=1.0 + 1e-15)
    return (x + (x.pow(2) - 1).clamp(min=0).sqrt()).clamp(min=1e-15).log()


def _tanh_clamp(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


# ---------- Poincaré Ball 流形 ----------
class PoincareBall:
    """Poincaré ball: ||x||^2 < 1/c，距离与映射用于 HGCN 与 Fermi-Dirac 解码。"""

    def __init__(self):
        self.min_norm = 1e-15
        self.eps = {torch.float32: 4e-3, torch.float64: 1e-5}

    def _lambda_x(self, x, c):
        x_sqnorm = x.pow(2).sum(dim=-1, keepdim=True)
        return 2 / (1.0 - c * x_sqnorm).clamp_min(self.min_norm)

    def proj(self, x, c):
        norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        maxnorm = (1 - self.eps[x.dtype]) / (c ** 0.5)
        cond = norm > maxnorm
        return torch.where(cond, x / norm * maxnorm, x)

    def proj_tan0(self, u, c):
        return u

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        return _tanh_clamp(sqrt_c * u_norm) * u / (sqrt_c * u_norm)

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = (1.0 / sqrt_c) * _artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def mobius_add(self, x, y, c, dim=-1):
        x2 = x.pow(2).sum(dim=dim, keepdim=True)
        y2 = y.pow(2).sum(dim=dim, keepdim=True)
        xy = (x * y).sum(dim=dim, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.min_norm)

    def mobius_matvec(self, m, x, c):
        sqrt_c = c ** 0.5
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        mx = x @ m.t()
        mx_norm = mx.norm(dim=-1, keepdim=True, p=2).clamp_min(self.min_norm)
        res_c = _tanh_clamp(mx_norm / x_norm * _artanh(sqrt_c * x_norm)) * mx / (mx_norm * sqrt_c)
        cond = (mx == 0).all(dim=-1, keepdim=True)
        res = torch.where(cond, torch.zeros_like(res_c), res_c)
        return res

    def dist(self, p1, p2, c):
        """双曲距离 d_L(z_i, z_j)，用于 Fermi-Dirac 解码。"""
        diff = self.mobius_add(-p1, p2, c, dim=-1)
        diff_norm = diff.norm(dim=-1, p=2, keepdim=False).clamp_min(self.min_norm)
        sqrt_c = c ** 0.5
        dist_c = _artanh(sqrt_c * diff_norm)
        return (2.0 / sqrt_c) * dist_c

    def dist_matrix(self, z, c):
        """批量计算 N×N 双曲距离矩阵，z: (N, d)。"""
        n = z.size(0)
        p1 = z.unsqueeze(1).expand(-1, n, -1)
        p2 = z.unsqueeze(0).expand(n, -1, -1)
        diff = self.mobius_add(-p1, p2, c, dim=-1)
        diff_norm = diff.norm(dim=-1, p=2).clamp_min(self.min_norm)
        sqrt_c = c ** 0.5
        return (2.0 / sqrt_c) * _artanh(sqrt_c * diff_norm)


# ---------- 双曲层 (edge_index 版，参考 hyp_layers) ----------
class HypLinear(nn.Module):
    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.dropout = dropout
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.bias)

    def forward(self, x):
        w = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(w, x, self.c)
        out = self.manifold.proj(mv, self.c)
        if self.bias is not None and self.bias.abs().sum() != 0:
            bias_tan = self.manifold.proj_tan0(self.bias.unsqueeze(0), self.c)
            hyp_bias = self.manifold.expmap0(bias_tan, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            out = self.manifold.proj(self.manifold.mobius_add(out, hyp_bias, self.c), self.c)
        return out


class HypAgg(nn.Module):
    """双曲邻域聚合，使用 edge_index。"""

    def __init__(self, manifold, c):
        super().__init__()
        self.manifold = manifold
        self.c = c

    def forward(self, x, edge_index, n):
        src, dst = edge_index[0], edge_index[1]
        x_tan = self.manifold.logmap0(x, c=self.c)
        ones = torch.ones(src.size(0), device=x.device, dtype=x.dtype)
        deg = scatter_add(ones, dst, dim_size=n).clamp(min=1).unsqueeze(-1)
        support = scatter_add(x_tan[src], dst, dim=0, dim_size=n) / deg
        return self.manifold.proj(self.manifold.expmap0(support, self.c), self.c)


class HypAct(nn.Module):
    def __init__(self, manifold, c_in, c_out, act=F.relu):
        super().__init__()
        self.manifold = manifold
        self.c_in, self.c_out = c_in, c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)


class HyperbolicGraphConvolution(nn.Module):
    """单层 HGCN：HypLinear -> HypAgg(edge_index) -> HypAct。"""

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout=0.0, act=F.relu, use_bias=True):
        super().__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)

    def forward(self, x, edge_index, n):
        h = self.linear(x)
        h = self.agg(h, edge_index, n)
        return self.hyp_act(h)


class HGCNEncoder(nn.Module):
    """
    3 层 HGCN 编码器：输入欧氏特征 x，输出双曲空间嵌入。
    输入 x (N, in_dim), edge_index (2, E)；输出 h (N, hidden) 在 Poincaré ball 上。
    """

    def __init__(self, in_dim, hidden_dim, c=1.0, dropout=0.0, num_layers=3):
        super().__init__()
        self.manifold = PoincareBall()
        self.c = c
        self.num_layers = num_layers
        curvatures = [c] * (num_layers + 1)
        dims = [in_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(HyperbolicGraphConvolution(
                self.manifold, dims[i], dims[i + 1],
                curvatures[i], curvatures[i + 1], dropout, F.relu, True
            ))

    def forward(self, x, edge_index, n):
        x_tan = self.manifold.proj_tan0(x, self.c)
        x_hyp = self.manifold.proj(self.manifold.expmap0(x_tan, self.c), self.c)
        for layer in self.layers:
            x_hyp = layer(x_hyp, edge_index, n)
        return x_hyp
