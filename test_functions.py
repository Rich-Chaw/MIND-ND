from utils.validate import validate_one_graph
num_features = 16
num_heads = 4
num_mps = 6
device = "cuda:0"
from networks.dismantle import load_sac_dismantler
import igraph as ig
graph = ig.Graph.Erdos_Renyi(100, 0.1)
graph['name'] = 'test'
policy, _, _, _, _ = load_sac_dismantler(num_features, num_heads, num_mps, 'mind', device)

validate_one_graph(graph, policy, step_ratio=0.0)