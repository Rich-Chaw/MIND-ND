import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn import GraphNorm
from torch_geometric.utils import degree

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.graph_data import Batch, Graph
from .gnn_utils import _get_init_features, _read_out


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        q = q * self.scale
        x = torch.matmul(q, k)
        if attn_bias is not None:
            x = x + attn_bias
        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)

        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads):
        super().__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class NAGphormerEncoder(nn.Module):
    """
    NAGphormer-like encoder adapted to MIND-ND interface:
    - Use K-hop token sequence per node (K = hops)
    - Apply Transformer encoder on token sequence
    - Use center-neighbor attention aggregation in original spirit
    - Return profile tensor for unified `_read_out` contract
    """

    def __init__(
        self,
        num_features,
        num_heads,
        num_mps,
        positional_encoding=None,
        handcrafted_features=False,
        **kwargs,
    ):
        super().__init__()
        self.num_features = 5 if handcrafted_features else num_features
        self.num_mps = num_mps
        self.positional_encoding = positional_encoding
        self.handcrafted_features = handcrafted_features

        # Keep unified GNN constructor interface unchanged:
        # (num_features, num_heads, num_mps, ...)
        # but use original NAGphormer defaults for internal transformer parameters.
        # Original defaults from NAGphormer/train.py:
        # hops=7, n_layers=1, n_heads=8, hidden_dim=512, dropout=0.1, attention_dropout=0.1.
        self.hops = int(kwargs.get("hops", 7))
        self.n_layers = int(kwargs.get("n_layers", 1))
        self.nag_num_heads = int(kwargs.get("n_heads", 8))
        self.hidden_dim = int(kwargs.get("hidden_dim", 512))
        self.dropout_rate = float(kwargs.get("dropout", 0.1))
        self.attention_dropout_rate = float(kwargs.get("attention_dropout", 0.1))
        if self.hops <= 0:
            raise ValueError(f"hops must be positive, got {self.hops}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        self.seq_len = self.hops + 1
        self.token_proj = nn.Linear(self.num_features, self.hidden_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    self.hidden_dim,
                    2 * self.hidden_dim,
                    self.dropout_rate,
                    self.attention_dropout_rate,
                    self.nag_num_heads,
                )
                for _ in range(self.n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.num_features)
        self.profile_act = nn.ReLU()
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
        self.graph_norm = GraphNorm(self.num_features * self.num_mps, eps=1e-4)
        self.apply(lambda module: init_params(module, n_layers=self.n_layers))

    def _build_transition_matrix(self, edge_index, num_nodes, dtype, device):
        src, dst = edge_index
        deg = degree(dst, num_nodes, dtype=dtype).clamp(min=1.0)
        vals = 1.0 / deg[dst]
        idx = torch.stack([dst, src], dim=0)
        return torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes), device=device).coalesce()

    def _build_hop_tokens(self, x0, edge_index):
        # Vectorized re_features-style propagation:
        # X^{k+1} = A_norm X^k, then stack [X^0, X^1, ..., X^K].
        trans = self._build_transition_matrix(edge_index, x0.size(0), x0.dtype, x0.device)
        token_list = [x0]
        x = x0
        for _ in range(self.hops):
            x = torch.sparse.mm(trans, x)
            token_list.append(x)
        return torch.stack(token_list, dim=1)  # (N, hops+1, F)

    def forward(self, g: Batch):
        x0 = _get_init_features(g, self.num_features, self.positional_encoding, self.handcrafted_features)
        tokens = self._build_hop_tokens(x0, g.edge_index)
        tokens = self.token_proj(tokens)

        for layer in self.layers:
            tokens = layer(tokens)
        output = self.final_ln(tokens)

        node_token = output[:, 0:1, :]
        hop_tokens = output[:, 1:, :]

        target = node_token.expand(-1, self.hops, -1)
        hop_scores = self.attn_layer(torch.cat((target, hop_tokens), dim=-1)).squeeze(-1)  # (N, hops)

        x_profile = torch.empty(g.total_nodes, self.num_features * self.num_mps, device=g.device)
        for k in range(self.num_mps):
            upto = min(k + 1, self.hops)
            weights_k = torch.softmax(hop_scores[:, :upto], dim=1).unsqueeze(-1)
            hop_ctx_k = torch.sum(hop_tokens[:, :upto, :] * weights_k, dim=1)
            x_k = (node_token.squeeze(1) + hop_ctx_k)
            x_k = self.profile_act(self.out_proj(x_k))
            x_profile[:, k * self.num_features : (k + 1) * self.num_features] = x_k

        x_profile = self.graph_norm(x_profile, g.batch)
        return _read_out(x_profile, g)


if __name__ == "__main__":
    device = torch.device("cpu")
    edge_index1 = np.array([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=np.int64)
    edge_index2 = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=np.int64)
    g1, g2 = Graph(edge_index1, 3), Graph(edge_index2, 4)
    batch = Batch(device, [g1, g2])
    model = NAGphormerEncoder(num_features=16, num_heads=4, num_mps=3).to(device)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (7, 2 * 16 * 3), f"Unexpected output shape: {out.shape}"
    print("NAGphormerEncoder smoke test passed:", out.shape)
