
import igraph as ig

# COPY from /code/testReal.py
def FINDER(graph,step_ratio=None,model_path='FINDER/models/graphSage_BA_nrange_30_50_m_4'):
    import sys
    import os
    FINDER_dir = "O:\My_Codes\GD2026\AAA-NetDQN\code"
    FINDER_type = model_path.split("/")[0]
    sys.path.append(FINDER_dir)
    sys.path.append(os.path.join(FINDER_dir,f'{FINDER_type}'))
    print(sys.path)
    if "moe" in FINDER_type:
        sys.path.append(os.path.join(FINDER_dir,'FINDER'))
        from GraphDQN import GraphDQN
        from MoEGraphDQN import MoEGraphDQN
        DQN_cls = MoEGraphDQN
    elif "advance" in FINDER_type:
        from AdvanceGraphDQN import AdvanceGraphDQN
        DQN_cls = AdvanceGraphDQN
    else:
        from GraphDQN import GraphDQN
        DQN_cls = GraphDQN
    
g_test = ig.Graph([(0,1),(0,2),(0,3),(1,3),(2,4),(3,4)])
FINDER(g_test)
