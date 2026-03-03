#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python interface for MIND-ND network dismantling
Usage: python python_interface.py --graph_file <path_to_graph> [--model_path <path>] [--threshold <threshold>]
"""

import sys
import os
import argparse
import json
import pickle
import tempfile
import igraph as ig
import numpy as np
import pathlib
import torch
from copy import deepcopy

# Add MIND-ND directory to path
sys.path.insert(0, os.path.dirname(__file__))
import utils
from networks.dismantle import load_sac_dismantler,load_sac_dismantler_with_film

def ensure_attribute(graph):
    """Ensure graph has required attributes for MIND-ND"""
    if 'static_id' not in graph.vs.attributes():
        graph.vs['static_id'] = list(range(graph.vcount()))
    
    if 'n_init' not in graph.attributes():
        graph['n_init'] = graph.vcount()
    
    return graph

def dismantle_graph(graph:ig.Graph,model_path=None,gnn='mind',**kwargs):
    '''
    same with mind_wrapper
    removals: complete solution
    removed_sizes: cumulative removed number at each step, start from 0
    lcc_sizes: lcc size at each step, len(lcc_sizes) = len(removed_sizes)
    '''
    graph = deepcopy(graph)
    ensure_attribute(graph)

    # device: str='cuda:0'
    device = torch.device('cpu')
    
    gnn = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    info = os.path.basename(os.path.dirname(model_path))
    print(info)

    F = 16; H = 4; K = 6
    latent_dim = 16
    ckpt = torch.load(model_path, map_location=device)

    from utils.validate import validate_one_graph
    if "task" in info:
        task_encoder = TaskEncoder(F, H, K, gnn, hidden_dim=128, latent_dim=latent_dim).to(device)
        if "task_encoder_state_dict" in ckpt:
            task_encoder.load_state_dict(ckpt["task_encoder_state_dict"])
        if "film" in info:
            policy, _, _, _, _, _ = load_sac_dismantler_with_film(F, H, K, gnn, device, latent_dim, model_path)

    else:
        policy, _, _, _, _ = load_sac_dismantler(F, H, K, gnn, device, model_path)
    auc, robustness, _, removals = validate_one_graph(graph,policy=policy,log_removals=True)
    
    return auc, robustness, removals

def main():
    parser = argparse.ArgumentParser(description='MIND-ND network dismantling interface')
    parser.add_argument('--graph_file', type=str, default="./graphs/simple/simple_100.pkl", help='Path to graph file')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model directory e.g. "./saved/mind/mind.ckpt')
    parser.add_argument('--threshold', type=float, default=0.1, help='threshold for dismantling (fraction of nodes)')
    parser.add_argument('--max_steps', type=int, default=None, help='Maximum number of steps')
    parser.add_argument('--out_file', type=str, default=None, help='Output path')
    
    args = parser.parse_args()
    
    # args.model_path = "./saved/hgnn_v4/sac_20260222_224028/10999.ckpt"


    # Load graph
    graph = None
    if args.graph_file:
        suffix = pathlib.Path(args.graph_file).suffix
        if suffix == '.pkl':
            with open(args.graph_file, 'rb') as f:
                graph = pickle.load(f)
                # Ensure it's an igraph object
                if not isinstance(graph, ig.Graph):
                    print("Error: Pickle file does not contain an igraph.Graph object", file=sys.stderr)
                    return 1
        elif suffix == '.txt':
            # Load edgelist and convert to igraph
            edges = []
            with open(args.graph_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        edges.append((int(parts[0]), int(parts[1])))
            graph = ig.Graph(edges=edges)
        elif suffix == '.gml':
            graph = ig.Graph.Read_GML(args.graph_file)
    
    if graph is None:
        print("Error: Failed to load graph", file=sys.stderr)
        return 1

    graph['name'] = args.graph_file
    
    # Dismantle graph
    auc, robustness, removals = dismantle_graph(
        graph, 
        model_path=args.model_path,
        threshold=args.threshold,
        max_steps=args.max_steps
    )
    
    # Output results
    result = {
        'removals': removals,
        'robustness': robustness,
        'num_nodes': graph.vcount(),
        'num_edges': graph.ecount()
    }
    
    if args.out_file:
        with open(args.out_file, 'w') as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
