import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import pdist, squareform

def random_graph(N,p):
    return ig.Graph.Erdos_Renyi(n=N, p=p)

def random_geometric_graph(N, r):
    return ig.Graph.GRG(N, r)

def random_regular_graph(N, k):
    """
    Generate a random k-regular graph using configuration model approach
    
    Args:
        N: Number of nodes
        k: Degree of each node (must result in even total degree)
    
    Returns:
        igraph Graph object
    """
    # Check if k-regular graph is possible
    if N * k % 2 != 0:
        raise ValueError("N * k must be even for a k-regular graph")
    
    if k >= N:
        raise ValueError("k must be less than N")
 
    # Use igraph's built-in regular graph generator if available
    try:
        g = ig.Graph.K_Regular(N, k)
        return g
    except:
        # Fallback to configuration model approach
        return _ring_based_regular(N, k)

def _ring_based_regular(N, k):
    """
    Ring-based construction for regular graphs (same as Watts-Strogatz with p=0)
    """
    if k % 2 != 0:
        raise ValueError("k must be even for ring-based construction")
    
    # Start with regular ring lattice
    g = ig.Graph.Ring(N, directed=False)
    
    # Add additional edges to reach k neighbors
    for i in range(2, k//2 + 1):
        for node in range(N):
            neighbor = (node + i) % N
            if not g.are_connected(node, neighbor):
                g.add_edge(node, neighbor)

    return g


def watts_strogatz_model(N, k, p):
    """
    Generate a Watts-Strogatz (WS) small-world model
    
    Args:
        N: Number of nodes
        k: Each node is connected to k nearest neighbors in ring topology
        p: Probability of rewiring each edge
    
    Returns:
        igraph Graph object
    """
    # Use igraph's built-in Watts-Strogatz model
    # Note: igraph uses 'nei' parameter for number of neighbors on each side
    # So k neighbors total means nei = k//2
    nei = max(1, k // 2)
    g = ig.Graph.Watts_Strogatz(1, N, nei, p)
    return g


def watts_strogatz_custom(N, k, p):
    """
    Custom implementation of Watts-Strogatz model for more control
    
    Args:
        N: Number of nodes
        k: Each node is connected to k nearest neighbors (must be even)
        p: Probability of rewiring each edge
    
    Returns:
        igraph Graph object
    """
    if k % 2 != 0:
        k += 1  # Make k even
    
    # Start with regular ring lattice
    g = ig.Graph.Ring(N, directed=False)
    
    # Add additional edges to reach k neighbors
    for i in range(2, k//2 + 1):
        for node in range(N):
            neighbor = (node + i) % N
            if not g.are_connected(node, neighbor):
                g.add_edge(node, neighbor)
    
    # Rewire edges with probability p
    edges_to_rewire = []
    for edge in g.get_edgelist():
        if np.random.rand() < p:
            edges_to_rewire.append(edge)
    
    for edge in edges_to_rewire:
        # Remove old edge
        g.delete_edges([edge])
        
        # Add new random edge (avoid self-loops and multiple edges)
        node1 = edge[0]
        attempts = 0
        while attempts < 100:  # Prevent infinite loops
            node2 = np.random.randint(N)
            if node2 != node1 and not g.are_connected(node1, node2):
                g.add_edge(node1, node2)
                break
            attempts += 1
    
    return g

def configuration_model(degrees):
    return ig.Graph.Degree_Sequence(degrees, method="vl")

def powerlaw(N,m,gamma):
    return ig.Graph.Static_Power_Law(n=N, m=m, exponent_out=gamma)

def LPA(N, m, gamma):
    g = ig.Graph(n=m+1)
    for nidx in range(g.vcount()):
        g.add_edges([(nidx, i) for i in range(nidx+1, g.vcount()) if i != nidx])
    a = m * (gamma - 3)
    for nidx in range(N - m - 1):
        node_count = g.vcount()
        node_weights = [g.degree(i) + a for i in range(node_count)]
        if np.sum(node_weights) == 0:
            node_weights = np.ones(node_count, dtype=float)
        node_weights /= np.sum(node_weights)
        end_nodes = np.random.choice(np.arange(node_count), m, p=node_weights, replace=False)
        g.add_vertex()
        g.add_edges([(node_count, i) for i in end_nodes])
    return g


def copying_model(N, m, gamma):
    g = ig.Graph(n=m+1)
    for nidx in range(g.vcount()):
        g.add_edges([(nidx, i) for i in range(nidx+1, g.vcount()) if i != nidx])
    alpha = (2 - gamma) / (1 - gamma)
    if not 0 < alpha < 1:
        raise Exception("Alpha needs to be between 0 and 1")
    for nidx in range(m + 1, N):
        g.add_vertex()
        for stub in range(m):
            if np.random.rand() < alpha:
                while True:
                    rand_endpoint = np.random.randint(nidx)
                    if not g.are_adjacent(nidx, rand_endpoint):
                        g.add_edge(nidx, rand_endpoint)
                        break
            else:
                while True:
                    rand_node = np.random.randint(nidx)
                    rand_endpoint = np.random.choice(g.neighbors(rand_node))
                    if not g.are_adjacent(nidx, rand_endpoint):
                        g.add_edge(nidx, rand_endpoint)
                        break
    return g


def stochastic_block_model(N,p_in,p_out,num_blocks=None):    
    """Generate a Stochastic Block Model (SBM) using igraph's built-in SBM function"""
    if num_blocks is None:
        num_blocks = np.random.randint(2, 5)  # 2-4 blocks
    
    # Simple equal block sizes
    n_nodes = [N // num_blocks] * num_blocks
    n_nodes[-1] += N % num_blocks  # Add remainder to last block
    
    pref_matrix = np.full((num_blocks, num_blocks), p_out)
    np.fill_diagonal(pref_matrix, p_in)
    
    # Generate SBM
    g = ig.Graph.SBM(sum(n_nodes), pref_matrix, n_nodes, directed=False, loops=False)
    
    return g
    # TO DELEBLOP: Herichcial SBM

def herichcial_stochastic_block_model(N,num_blocks=None):    
    """Generate a Stochastic Block Model (SBM) using igraph's built-in SBM function"""
    if num_blocks is None:
        num_blocks = np.random.randint(2, 5)  # 2-4 blocks
    
    # Simple equal block sizes
    n_nodes = [N // num_blocks] * num_blocks
    n_nodes[-1] += N % num_blocks  # Add remainder to last block
    
        # Use ratio: p_in should be 5-10x higher than p_out for good modularity
    ratio = 5 * np.random.uniform(1,6) # 5-10x
    
    p_out = np.random.uniform(0.01,0.1)
    p_in = ratio*p_out

    pref_matrix = np.full((num_blocks, num_blocks), p_out)
    np.fill_diagonal(pref_matrix, p_in)
    
    # Generate SBM
    g = ig.Graph.SBM(sum(n_nodes), pref_matrix, n_nodes, directed=False, loops=False)
    
    return g
    # TO DELEBLOP: Herichcial SBM

def corrupting(g):
    """
    Network augmentation through edge corruption
    
    Args:
        g: igraph Graph object
        noise_type: 0 for edge deletion, 1 for edge addition
        n_coeff: ratio of edges to be corrupted (0.0 to 1.0)
    
    Returns:
        Modified graph
    """
    noise_type = np.random.randint(2)  # 0: delete edges, 1: add edges
    n_coeff = np.random.choice([0.01,0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    print('noise_type =', noise_type, ', n_coeff =', n_coeff)

    if n_coeff <= 0:
        return g
    
    N = g.vcount()
    current_edges = g.ecount()
    
    if noise_type == 0:  # Edge deletion
        if current_edges == 0:
            return g
        
        # Calculate number of edges to delete
        edges_to_delete = max(1, int(current_edges * n_coeff))
        edges_to_delete = min(edges_to_delete, current_edges - N + 1)  # Keep graph connected
        
        # Ensure we don't try to delete more edges than available
        if edges_to_delete <= 0:
            return g
        
        # Randomly select edges to delete
        edge_list = list(g.get_edgelist())
        edges_to_delete = min(edges_to_delete, len(edge_list))
        
        if edges_to_delete > 0:
            edges_to_remove = np.random.choice(len(edge_list), edges_to_delete, replace=False)
            
            # Delete edges (in reverse order to maintain indices)
            for idx in sorted(edges_to_remove, reverse=True):
                g.delete_edges([edge_list[idx]])
            
    elif noise_type == 1:  # Edge addition
        # Calculate maximum possible edges
        max_edges = N * (N - 1) // 2
        possible_new_edges = max_edges - current_edges
        
        if possible_new_edges == 0:
            return g
        
        # Calculate number of edges to add
        edges_to_add = max(1, int(current_edges * n_coeff))
        edges_to_add = min(edges_to_add, possible_new_edges)
        
        # Find all possible edges that don't exist
        existing_edges = set(g.get_edgelist())
        existing_edges.update([(j, i) for i, j in existing_edges])  # Add reverse edges
        
        possible_edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) not in existing_edges:
                    possible_edges.append((i, j))
        
        if len(possible_edges) == 0:
            return g
        
        # Randomly select edges to add
        edges_to_add = min(edges_to_add, len(possible_edges))
        
        if edges_to_add > 0:
            new_edges = np.random.choice(len(possible_edges), edges_to_add, replace=False)
            
            # Add new edges
            for idx in new_edges:
                g.add_edge(possible_edges[idx][0], possible_edges[idx][1])
    
    return g


def preprocess(g, target_min=200, target_max=300):
    """
    Clean network: remove isolated nodes, merge/trim to target size range
    
    Args:
        g: igraph Graph object
        target_min/max: Size range for final connected component
    
    Returns:
        Processed igraph Graph with single component in target size range
    """
    if g.vcount() == 0:
        return g
    
    # Step 1: Remove isolated nodes and get largest components
    components = [c for c in g.connected_components() if len(c) > 1]
    if not components:
        return ig.Graph(n=0)  # Return empty graph if no valid components
    
    components.sort(key=len, reverse=True)
    
    # Step 2: Merge components if largest is too small
    if len(components[0]) < target_min:
        # Combine top components until we reach target_min
        merged_nodes = []
        for comp in components:
            merged_nodes.extend(comp)
            if len(merged_nodes) >= target_min:
                break
        if len(merged_nodes) < target_min:
            print("the graph is too isolated...")
            return ig.Graph(n=2)

        # Create subgraph and connect components
        g = g.subgraph(merged_nodes)

        # Connect all components in graph with minimal edges
        components = g.connected_components()
        for i in range(len(components) - 1):
            # Connect adjacent components with single edge
            node1 = np.random.choice(components[i])
            node2 = np.random.choice(components[i + 1])
            g.add_edge(node1, node2)
    else:
        # Use only the largest component
        g = g.subgraph(components[0])
    
    return g

def switch(g, order, type):
    swt_trials = 0
    while True:
        swt_trials += 1
        if swt_trials > 100:
            return False
        e1, e2 = np.array(g.get_edgelist())[np.random.choice(g.ecount(), 2, replace=False)]
        rand_idx = np.random.randint(2, size=2)
        i, l = e1[rand_idx[0]], e1[1 - rand_idx[0]]
        j, k = e2[rand_idx[1]], e2[1 - rand_idx[1]]
        if g.are_adjacent(i, k) or g.are_adjacent(j, l) or len(list({i, j, k, l})) < 4:
            continue
        if (order[i] - order[j]) * (order[k] - order[l]) * type >= 0:
            g.delete_edges([tuple(e1), tuple(e2)])
            g.add_edges([(i, k), (j, l)])
            break
    return g

def rewiring(g,switch_type,r_coeff):
    ordering = 'deg' if np.random.rand() <= 0.5 else 'rnd'
    node_order = np.random.permutation(g.vcount()) if ordering == 'rnd' else g.degree()
    switch_no = 0
    while True:
        g = switch(g, node_order, type=switch_type)
        switch_no += 1
        if g is False:
            print('Switching trials maxed out; generating new net...')
            g = ig.Graph(n=2)
            break
        if (switch_type * g.assortativity(node_order) > switch_type * r_coeff) or \
            (switch_type == 0 and np.abs(g.assortativity(node_order)) < r_coeff):
                break
        if switch_no > 100000:
            print('Taking too many switches...')
            g = ig.Graph(n=2)
            break
    return g, node_order, ordering, switch_no

def targeted_rewiring(g, target_q, target_r, iterations=1000):
    current_q = g.community_leiden(objective_function="modularity").modularity
    current_r = graph.assortativity_degree()
    
    for _ in range(iterations):
        # Perform a degree-preserving swap
        # (Standard X-swap: pick two edges (a,b) and (c,d) -> (a,d) and (c,b))
        test_graph = graph.copy()
        test_graph.rewire(n=1) 
        
        new_q = test_graph.community_leiden(objective_function="modularity").modularity
        new_rt = test_graph.assortativity_degree()
        
        # Calculate Euclidean distance to the "blank spot" on your scatter plot
        old_dist = ((current_q - target_q)**2 + (current_r - target_r)**2)**0.5
        new_dist = ((new_q - target_q)**2 + (new_r - target_r)**2)**0.5
        
        if new_dist < old_dist:
            graph = test_graph
            current_q, current_r = new_q, new_r
            
    return graph

def community_aware_rewiring(g,K=5,iterations=1000,mode='increase'):
    """
    Rewires a graph to change modularity while strictly preserving degree distribution.
    
    Args:
        g: igraph.Graph object
        K: Number of communities to randomly assign
        iterations: Number of swap attempts
        mode: 'increase' to boost modularity, 'decrease' to reduce it
    """
    # 1. Randomly assign nodes to K groups (clusters)
    membership = np.random.randint(0, K, size=g.vcount())
    
    # Define the objective: 1 if endpoints are in the same group, 0 otherwise
    def is_intra(u, v):
        return 1 if membership[u] == membership[v] else 0

    for _ in range(iterations):
        if g.ecount() < 2:
            break
            
        # 2. Pick two random edges (u, v) and (x, y)
        e1_idx, e2_idx = np.random.choice(g.ecount(), 2, replace=False)
        u, v = g.es[e1_idx].tuple
        x, y = g.es[e2_idx].tuple

        # 3. Ensure all 4 nodes are distinct to preserve degrees and avoid loops
        if len({u, v, x, y}) < 4:
            continue
            
        # 4. Check if proposed edges (u, x) and (v, y) already exist
        if g.are_adjacent(u, x) or g.are_adjacent(v, y):
            continue

        # 5. Calculate 'Intra-community' edge count before and after
        # Modularity is increased by maximizing intra-community edges
        old_score = is_intra(u, v) + is_intra(x, y)
        new_score = is_intra(u, x) + is_intra(v, y)

        # 6. Accept swap based on mode
        accept = False
        if mode == 'increase' and new_score > old_score:
            accept = True
        elif mode == 'decrease' and new_score < old_score:
            accept = True
        elif new_score == old_score and np.random.rand() < 0.1: 
            # Allow neutral swaps with small probability to maintain randomness
            accept = True

        if accept:
            g.delete_edges([(u, v), (x, y)])
            g.add_edges([(u, x), (v, y)])
            
    return g

def triangle_favor_rewiring(g, iterations=100, temperature=0.01):
    """
    Performs degree-preserving swaps that favor the creation of triangles.
    
    Args:
        g: The igraph object to rewire.
        iterations: Number of swap attempts.
        temperature: Controls the probability of accepting a 'worse' swap
                     (higher = more random exploration).
    """
    # Create sets of neighbors for fast intersection
    # This is much faster than calling g.neighbors() inside the loop
    adj_sets = [set(g.neighbors(v)) for v in range(g.vcount())]
    
    for _ in range(iterations):
        # 1. Pick two random edges (u, v) and (x, y)
        e1_idx, e2_idx = np.random.choice(g.ecount(), 2, replace=False)
        u, v = g.es[e1_idx].tuple
        x, y = g.es[e2_idx].tuple

        # 2. Ensure all 4 nodes are distinct to strictly preserve degree
        if len({u, v, x, y}) < 4:
            continue
            
        # 3. Check if proposed edges (u, x) and (v, y) already exist
        if g.are_adjacent(u, x) or g.are_adjacent(v, y):
            continue

        # 4. Calculate 'Local Triangle Count' before swap
        # Shared neighbors for current edges: (u,v) and (x,y)
        old_tri = len(adj_sets[u] & adj_sets[v]) + len(adj_sets[x] & adj_sets[y])

        # 5. Calculate potential 'Local Triangle Count' after swap
        # Shared neighbors for proposed edges: (u,x) and (v,y)
        new_tri = len(adj_sets[u] & adj_sets[x]) + len(adj_sets[v] & adj_sets[y])

        # 6. Metropolis-Hastings Acceptance
        # Accept if it increases triangles, or with a small probability if it decreases
        delta = new_tri - old_tri
        if delta >= 0 or np.random.rand() < np.exp(delta / temperature):
            # Update Graph
            g.delete_edges([(u, v), (x, y)])
            g.add_edges([(u, x), (v, y)])
            
            # Update our local adjacency sets to keep them in sync
            adj_sets[u].remove(v); adj_sets[u].add(x)
            adj_sets[v].remove(u); adj_sets[v].add(y)
            adj_sets[x].remove(y); adj_sets[x].add(u)
            adj_sets[y].remove(x); adj_sets[y].add(v)
            
    return g


if __name__ == '__main__':
    net_dict = {}
    for net_no in range(10000):
        # topology = np.random.choice(['LPA', 'Copy', 'ER', 'SBM', 'DC-SBM', 'RGG'])
        topology = np.random.choice(['LPA','Copy','ER'])
        N = 100 + np.random.randint(101)
        gamma = 2.5 + np.random.rand()
        switch_type = np.random.randint(3) - 1 #[-1,0,1]
        r_coeff = 0.05 if switch_type == 0 \
                else switch_type * np.random.choice([0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
        
        m = np.random.choice([1, 2, 3, 4, 5, 6, 8, 10],
                        p=[1/12, 2/12, 2/12, 2/12, 2/12, 1/12, 1/12, 1/12])


        if topology in ['RGG']:
            r = np.sqrt(np.log(N) / (np.pi * N))
            r = np.random.uniform(1.1,2.0) * r


        print(net_no, ':', topology, ', N =', N, ', m =', m,'gamma = ', gamma,'r_target =', r_coeff, 'switch_type = ', switch_type)

        trial = 0
        while True:
            if trial > 100:
                print('Network regeneration trials maxed out, increasing m to', m + 1)
                trial = 0
                m += 1

            if topology == 'ER':
                p = ((N - 1) * m - 1) / (N * (N - 1))
                net = ig.Graph.Erdos_Renyi(n=N, p=p)
            elif topology == 'Copy':
                net = copying_model(N, m, gamma)
                if m==1:
                    r_coeff = 0.01 + 0.04 * np.random.rand() if switch_type == 0 else \
                                        switch_type * np.random.choice([0.05, 0.1, 0.15])
            elif topology == 'LPA':
                net = LPA(N, m, gamma)
                if m==1:
                    r_coeff = 0.01 + 0.04 * np.random.rand() if switch_type == 0 else \
                                        switch_type * np.random.choice([0.05, 0.1, 0.15])
            elif topology == 'WS':
                p = np.random.choice([0.05,0.1,0.15,0.2])
                watts_strogatz_model(N, k=m, p=p)
            elif topology == 'SBM':
                net = stochastic_block_model(N, m)
            elif topology == 'RGG':
                r = np.sqrt(np.log(N) / (np.pi * N))
                r = np.random.uniform(1.1,2.0) * r
                net = random_geometric_graph(N, r=r)
            else:
                raise Exception('Topology not valid!!')
            
            # Process network: delete isolated nodes, merge top components, ensure size constraints
            # net = preprocess(net, target_size_min=200, target_size_max=300)
            
            # Optional: Enhance modularity by adding triangles before assortativity switching
            # if np.random.rand() < 0.3: # Apply to 30% of graphs to fill high-modularity gaps
            #     net = triangle_favor_rewiring(net, iterations=500)

            # if np.random.rand() < 0.8: # Apply to 80% of the dataset
            #     target_k = np.random.randint(2, 10)
            #     # Randomly choose to increase or decrease community structure
            #     mode = 'increase' if np.random.rand() > 0.2 else 'decrease'
            #     net = community_aware_rewiring(net, K=target_k, iterations=N*5, mode=mode)

            # if net.vcount() > 2:
            #     net, node_order, ordering, switch_count = rewiring(net,switch_type,r_coeff)

            # Corruption/augmentation
            # if net.vcount() > 2:  # Only apply corruption if we have a valid network
            #     net = corrupting(net)

            if net.is_connected():
                # print('r_final =', net.assortativity(node_order), ordering)
                net_dict[net_no] = {'adj': np.array(net.get_adjacency().data, dtype=bool)}
                net_dict[net_no]['info'] = {
                    'topology': topology,
                    'size': N,
                    'mean_deg': np.mean(net.degree()),
                    # 'assortativity': net.assortativity(node_order),
                    # 'ordering': ordering,
                    # 'switch_count': switch_count
                }
                if topology in ['LPA', 'Copy']:
                    net_dict[net_no]['info']['gamma'] = gamma
                if net_no % 100 == 0:
                    with open('switched_graphs.pkl', 'wb') as out_f:
                        pickle.dump(net_dict, out_f)
                break
            trial += 1

