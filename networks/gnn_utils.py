import torch
from utils.graph_data import Batch


def _get_init_features(g: Batch, num_features, positional_encoding, handcrafted_features=False):
    """Get initial node features. Device is taken from g.edge_index.
    When handcrafted_features: returns (N, 5); else RW or all-ones (N, num_features).
    RW uses random-walk return probabilities only (same width as num_features; do not concat with ones).
    """
    device = g.device
    if hasattr(g, "x_init") and g.x_init is not None:
        x_init = g.x_init
        if x_init.dim() == 2 and x_init.shape[0] == g.total_nodes and x_init.shape[1] == num_features:
            return x_init.to(device=device, dtype=torch.float32)

    if handcrafted_features:
        from utils.graph_data import handcrafted_node_features
        x_init =  handcrafted_node_features(g, device)
    elif positional_encoding == 'RW':
        from utils.graph_data import random_walk_positional_encoding
        x_init = random_walk_positional_encoding(g, num_features, device)
    else:
        x_init = torch.ones(g.total_nodes, num_features, device=device, dtype=torch.float32)
    return x_init


def _read_out(x_profile, g:Batch):
    """Unified output: concat(node_embedding, graph_embedding) -> (N, 2KF)."""
    return torch.cat([
        x_profile[g.non_omni_mask],
        x_profile[g.omni_ids[g.batch_non_omni]]
    ], dim=1)
