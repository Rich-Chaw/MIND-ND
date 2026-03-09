"""
Hyperbolic GCN (HGCN) on Poincaré ball，参考 hgcn/models/encoders.py 与 layers/hyp_layers。
支持 edge_index 输入，用于图 VAE 编码器与双曲距离解码。
"""
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

    def to_hyperbolic(self, u, c):
        """把切空间向量映射回 Poincare ball，并做一次投影保证数值稳定。"""
        return self.proj(self.expmap0(u, c), c)

    def to_tangent(self, x, c):
        """把双曲点映射回原点切空间。"""
        return self.logmap0(x, c)

    def expmap0(self, u, c):
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        return _tanh_clamp(sqrt_c * u_norm) * u / (sqrt_c * u_norm)

    def logmap0(self, p, c):
        sqrt_c = c ** 0.5
        p_norm = p.norm(dim=-1, p=2, keepdim=True).clamp_min(self.min_norm)
        scale = (1.0 / sqrt_c) * _artanh(sqrt_c * p_norm) / p_norm
        return scale * p

    def log_det_exp0(self, u, c):
        """
        Wrapped Normal 在 Poincaré ball 上：exp_0 的 log-Jacobian 行列式。
        log p(z) = log p_T(v) - log_det(d exp_0(v)/dv)，用于双曲 KL 计算。
        公式：det = (1/cosh^2(sqrt_c*r)) * (tanh(sqrt_c*r)/(sqrt_c*r))^{d-1}，r=||u||。
        """
        sqrt_c = c ** 0.5
        r = u.norm(dim=-1, p=2).clamp_min(self.min_norm)
        sqrt_c_r = (sqrt_c * r).clamp(min=self.min_norm)
        # r=0 时 log_det -> 0（极限 tanh(x)/x -> 1）
        log_cosh = torch.log(torch.cosh(sqrt_c_r.clamp(max=12)))
        tanh_sqrt_c_r = _tanh_clamp(sqrt_c_r)
        log_tanh_over_r = torch.log(tanh_sqrt_c_r.clamp(min=1e-10)) - torch.log(sqrt_c_r.clamp(min=1e-10))
        d = u.size(-1)
        log_det = -2.0 * log_cosh + (d - 1) * log_tanh_over_r
        return log_det

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
        nn.init.xavier_uniform_(self.weight, gain=1.0)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        w = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(w, x, self.c)
        out = self.manifold.proj(mv, self.c)
        if self.bias is not None and self.bias.abs().sum() != 0:
            bias_tan = self.manifold.proj_tan0(self.bias.unsqueeze(0), self.c)
            hyp_bias = self.manifold.to_hyperbolic(bias_tan, self.c)
            out = self.manifold.proj(self.manifold.mobius_add(out, hyp_bias, self.c), self.c)
        return out


class HypAgg(nn.Module):
    """双曲邻域聚合，使用 edge_index。"""

    def __init__(self, manifold, c):
        super().__init__()
        self.manifold = manifold
        self.c = c

    def _add_self_loops(self, edge_index, n):
        """为每个节点添加自环，保证聚合时保留自身信息。"""
        loop_index = torch.arange(0, n, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        return torch.cat([edge_index, loop_index], dim=1)

    def _symmetric_norm(self, edge_index, n, device):
        """对称归一化系数: 1 / sqrt(deg_i * deg_j)。"""
        src, dst = edge_index[0], edge_index[1]
        ones = torch.ones(edge_index.size(1), device=device)
        deg = scatter_add(ones, dst, dim_size=n).clamp(min=1)
        return src, dst, torch.pow(deg[src] * deg[dst], -0.5).unsqueeze(-1)

    def forward(self, x, edge_index, n):
        edge_index = self._add_self_loops(edge_index, n)
        src, dst, norm = self._symmetric_norm(edge_index, n, x.device)
        x_tan = self.manifold.to_tangent(x, c=self.c)
        support = scatter_add(norm * x_tan[src], dst, dim=0, dim_size=n)
        return self.manifold.to_hyperbolic(support, self.c)


class HypAct(nn.Module):
    def __init__(self, manifold, c_in, c_out, act=F.relu):
        super().__init__()
        self.manifold = manifold
        self.c_in, self.c_out = c_in, c_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold.to_tangent(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.to_hyperbolic(xt, c=self.c_out)


class HyperbolicGraphConvolution(nn.Module):
    """单层 HGCN：HypLinear -> HypAgg(edge_index) -> HypAct。"""

    def __init__(self, manifold, in_features, out_features, c_in, c_out, dropout=0.0, act=F.leaky_relu, use_bias=True):
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

    def __init__(self, in_dim, hidden_dim, c=1.0, dropout=0.1, num_layers=2):
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
                curvatures[i], curvatures[i + 1], dropout, F.leaky_relu, True
            ))

    def forward(self, x, edge_index, n):
        # 对输入特征进行缩放，防止初始 expmap0 直接映射到双曲边缘
        x_tan = self.manifold.proj_tan0(x, self.c)
        x_hyp = self.manifold.to_hyperbolic(x_tan, self.c)
        for layer in self.layers:
            x_hyp = layer(x_hyp, edge_index, n)
        return x_hyp
