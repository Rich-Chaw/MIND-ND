# Algorithm: Multi-Agent MIND with Global Coordinator
Input: Global Graph G, K Partition Agents {A_1...A_K}, 1 Global Agent C

1:  While LCC(G) > target:
2:      # 1. Partitioning
3:      subgraphs = Partition(G, K)
4:      
5:      # 2. Global Coordinator computes Global Context
6:      # This reflects the LCC of the entire G, not just subgraphs
7:      Z_global = C.compute_global_omni_node(G) 
8:      
9:      # 3. Parallel Local Inference with Global Awareness
10:     Candidate_Nodes = []
11:     For i in 1 to K:
12:         # The Partition Agent combines local features with the Global Context
13:         # This ensures the agent knows if its partition is part of the LCC
14:         Probs_i = A_i.evaluate(subgraphs[i], Z_global)
15:         Candidate_Nodes.append(Probs_i)
16:         
17:     # 4. Consensus & Removal
18:     # The Global Coordinator picks the best node across all agents
19:     v_target = C.select_best_node(Candidate_Nodes)
20:     G = G \ {v_target}