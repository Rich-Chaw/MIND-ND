
import igraph as ig
import numpy as np
import sys
import pandas as pd
import os
import importlib

from scipy.integrate._ivp.radau import P
from utils import graph_models,structural_diversity_analysis
from utils.graph_models import *

def generate(topology,nrange = "100_200",num=10,is_rewiring=False):
    graphs = []
    n_min = int(nrange.split("_")[0])
    n_max = int(nrange.split("_")[1])
    idx = 0
    
    while idx < num:
        N = np.random.randint(n_min,n_max+1)
        if topology in ['Ring','BA','Copy','LPA','ER','HK','PLC','BTER']:
            m = int(np.random.choice([1, 2, 3, 4, 5, 6, 8, 10],
                                    p=[1/12, 2/12, 2/12, 2/12, 2/12, 1/12, 1/12, 1/12]))
        if topology in ['SBM','DCSBM','Barbell','StarCom','RingCom']:
            m = int(np.random.choice([3, 4, 5, 6, 8, 10, 12],
                                    p=[3/12, 3/12, 2/12, 2/12, 1/12, 1/24, 1/24]))
        
        if topology == 'Ring':
            g = ig.Graph.Ring(N, directed=False)
            config = {'N':N}
        elif topology == '2D-lattice':
            g = ig.Graph.Lattice([5, 5], nei=1, directed=False, mutual=False, circular=False)
            config = {'N':N}
        elif topology == 'Regular':
            g = random_regular_graph(N,k=m)
            config = {'N':N,'k':m}
        elif topology == 'WS':
            p = np.random.uniform(0.01,0.05)
            g = WS(N,k=m,p=0.2)
            config = {'N':N,'k':m,'p':float(p)}
        elif topology == 'ER':
            p=((N - 1) * m - 1) / (N * (N - 1))
            g = ig.Graph.Erdos_Renyi(N, p)
            config = {'N':N,'p':float(p)}
        elif topology == 'BA':
            g = ig.Graph.Barabasi(N, m, directed=False)
            config = {'N':N,'m':m}
        elif topology == 'Copy':
            gamma = 2.5 + np.random.rand()
            g = copying_model(N, m, gamma)
            config = {'N':N,'m':m,'gamma':float(gamma)}
        elif topology == 'LPA':
            gamma = 2.5 + np.random.rand()
            g = LPA(N, m, gamma)
            config = {'N':N,'m':m,'gamma':float(gamma)}
        elif topology == 'PLC':
            p = np.random.uniform(0.1,1.0)
            g = powerlaw_cluster(N,m,p)
            config = {'N':N,'m':m,'gamma':float(gamma),'p':p}
        elif topology == 'HK':
            p = np.random.uniform(0.1,1.0)
            g = holme_kim(N,m,p)
            config = {'N':N,'m':m,'p':p}
        elif topology == 'FF':
            p = np.random.uniform(0.2,0.4)
            r = np.random.uniform(0.4,0.6)
            g = forest_fire(N,p,r)
            config = {'N':N,'p':p,'r':r}
        elif topology == 'FFC':
            p = np.random.uniform(0.2,0.4)
            r = np.random.uniform(0.4,0.6)
            n_amb = random.randint(1,3)
            g = forest_fire_custom(N,p,r,n_amb)
            config = {'N':N,'p':p,'r':r,'n_amb':n_amb}
        elif topology == 'BTER':
            gamma = 2.5 + np.random.rand()
            rho = np.random.uniform(0.3,0.6)
            eta = np.random.uniform(0.5,1.5)
            g = BTER(N,gamma,rho,eta)
            config = {'N':N,'gamma':float(gamma),'rho':float(rho),'eta':float(eta)}
        elif topology == 'SBM':
            num_blocks = np.random.randint(2,5)
            p_in = np.random.uniform(0.1,0.15)
            p_out = np.random.uniform(0.001,0.006)
            g = SBM(N,p_in,p_out,num_blocks)
            config = {'N':N,'p_in':p_in,'p_out':p_out,'blocks':num_blocks}
        elif topology == 'DCSBM':
            num_blocks = np.random.randint(2,5)
            p_in = np.random.uniform(0.1,0.15)
            p_out = np.random.uniform(0.001,0.006)
            g = DCSBM(N,p_in,p_out,num_blocks)
            config = {'N':N,'p_in':p_in,'p_out':p_out,'blocks':num_blocks}
        elif topology == 'LFR':
            tau1 = 2.5 + np.random.rand()
            tau2 = np.random.uniform(1.0,2.0)
            mu = np.random.uniform(0.05,0.3)
            m = np.random.randint(3,12)
            min_comm = 10
            max_deg = int(N * 0.12)
            g = LFR(N,m,tau1,tau2,mu,min_comm,max_deg)
            if g is None:
                continue
            config = {'N':N,'m':m,'tau1':float(tau1),'tau2':float(tau2),'mu':float(mu),'min_comm':min_comm,'max_deg':max_deg}
        elif topology == 'RGG':
            r = np.sqrt(np.log(N) / (np.pi * N))
            r = np.random.uniform(1.1,2.0) * r
            g = random_geometric_graph(N, r)
            config = {'N':N,'r':float(r)}
        elif topology == "Barbell":
            p_in = min(2 * m / N,0.5)
            path_len = np.random.randint(1,5)
            config['N'] = N; config['p_in'] = p_in; config['len'] = path_len
            g = barbell(N,p_in,path_len)
        elif topology == "StarCom":
            num_communities = np.random.randint(3,8)
            p_in = min(num_communities * m / N, 0.5)
            g = star_community(N,p_in,num_communities)
            config = {'N':N,'p_in':p_in,'comms':num_communities}
        elif topology == "RingCom":
            num_communities = np.random.randint(3,8)
            p_in = min(num_communities * m / N, 0.5)
            g = ring_community(N,p_in,num_communities)
            config = {'N':N,'p_in':p_in,'comms':num_communities}
        elif topology == "Necklace":
            num_cliques = np.random.randint(5,10)
            g = necklace(N,num_cliques)
            config = {'N':N,'clqs':num_cliques}
        else:
            raise Exception('Topology not valid!!')
        config['topology'] = topology

        if g.is_connected():
            g = g  # Already connected, no need to extract
        else:
            g = g.subgraph(max(g.connected_components(), key=len)) 
        
        if is_rewiring:
            # switch_type = np.random.randint(3) - 1 #[-1,0,1]
            # r_coeff = 0.05 if switch_type == 0 \
            #         else switch_type * np.random.choice([0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
            # if topology in ['LPA','Copy']:
            #     if m==1:
            #         r_coeff = 0.01 + 0.04 * np.random.rand() if switch_type == 0 else \
            #                             switch_type * np.random.choice([0.05, 0.1, 0.15])
            # # print("r_target: ",r_coeff," switch_type: ",switch_type)
            
            # g, node_order, ordering, switch_count = rewiring(g,switch_type,r_coeff)
            # if g.is_connected():
            #     g = g  # Already connected, no need to extract
            # else:
            #     g = g.subgraph(max(g.connected_components(), key=len))

            g.rewire(n=100, mode="simple")

        
        
        if g.is_connected() and g.vcount() >= n_min and g.vcount() <= n_max:
            # Store config as graph attribute so it is saved with the graph (e.g. when pickling g)
            g["config"] = config
            graphs.append(g)
            idx += 1
   
    return graphs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="graphs/", help="The directory to save the graphs")
    parser.add_argument("--mode", type=str, default="train", help="The mode to run the script")
    parser.add_argument("--nrange", type=str, default="100_200", help="The range of the number of nodes")
    parser.add_argument("--num", type=int, default=1000, help="The number of graphs to generate")
    parser.add_argument("--with_label", type=bool, default=False, help="Whether to save the label")
    parser.add_argument("--rewire", action="store_true", help="Whether to rewire the graphs")
    args = parser.parse_args()
    

    # topologies = ['LFR']
    # topologies = ['LPA','Copy','ER']
    topologies = ['FF']
    # topologies = ['BTER']
    # ['SBM','DCSBM','LPA','Copy','ER']

    dataset_name = f"{args.nrange}"
    for topology in topologies:
        dataset_name += f"_{topology}"
    dataset_name += f"_{args.num}"
    if args.with_label:
        dataset_name += "_with_label"
    if args.rewire:
        dataset_name += "_rewire"
    
    save_dir = os.path.join(args.save_dir, args.mode, dataset_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving graphs to {save_dir}")

    graphs = []
    for file in os.listdir(save_dir):
        if file.endswith('.pkl'):
            with open(os.path.join(save_dir,file),'rb') as f:
                graphs.append(pickle.load(f))
    print(f"Found {len(graphs)} existing graphs")

    for g_idx in range(len(graphs),args.num):
        # topology = np.random.choice(['Barbell','StarCom','RingCom','SBM'])
        # topology = np.random.choice(['SBM','DCSBM','LPA','Copy','ER'])
        topology = np.random.choice(topologies)

        g = generate(topology,args.nrange,1,is_rewiring=args.rewire)[0]
        if args.with_label:
            save_name = f"{g_idx:05d}_{topology}.pkl"
        else:
            save_name = f"{g_idx:05d}.pkl"
        with open(os.path.join(save_dir,save_name),'wb') as f:
            pickle.dump(g,f)

        graphs.append(g)

    print(f"generated {len(graphs)} graphs")
    
    # net_dict = {}
    # for net_no, g in enumerate(graphs):
    #     # Config is also stored on the graph as g["config"]
    #     config = g['config']
    #     topology = config.get('topology')
    #     net_dict[net_no] = {'adj': np.array(g.get_adjacency().data, dtype=bool)}
    #     net_dict[net_no]['info'] = {
    #         'topology': topology,
    #         'size': g.vcount(),
    #         'mean_deg': np.mean(g.degree()),
    #         'assortativity': g.assortativity_degree(),
    #         'config': config,
    #     }
        

    from utils.structural_diversity_analysis import (
        calculate_properties,
        create_scatter_plot
    )
    df = calculate_properties(graphs)
    print(len(df))
    create_scatter_plot(df["Q"].values, df["r"].values, df["label"].values, os.path.join(save_dir, "diversity_scatter.png"))
    create_scatter_plot(df["Q"].values, df["clustering"].values, df["label"].values, os.path.join(save_dir, "diversity_scatter_QC.png"))
    # with open(f'{dataset_name}.pkl', 'wb') as out_f:
    #     pickle.dump(net_dict, out_f)