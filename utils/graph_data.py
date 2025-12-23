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

        self.act_offsets = torch.tensor(act_offsets, dtype=torch.long, device=self.device) # (B)

        self.omni_ids = torch.tensor(omni_ids, dtype=torch.long, device=self.device)  # (B)
        self.non_omni_mask = torch.ones(self.total_nodes, dtype=torch.bool, device=self.device) #(N+B)
        self.non_omni_mask[self.omni_ids] = False

        self.edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device)
        self.batch = torch.tensor(batch, dtype=torch.long, device=self.device) #(N+B)
        self.batch_non_omni = self.batch[self.non_omni_mask] #(N)