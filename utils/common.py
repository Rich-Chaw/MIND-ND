import pickle
import igraph as ig
import networkx as nx

def load_g(path, name): 
    g_type = path.split('.')[-1]
    if g_type == 'pkl':
        with open(path, "rb") as f:
            g = pickle.load(f)
        g.simplify(multiple=True, loops=True)
        g['name'] = name
        return g
    else: pass