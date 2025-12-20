import pandas as pd
def statistics(g):
    leiden_comm = g.community_leiden(
        objective_function="modularity", 
        weights=None, 
        resolution_parameter=1.0, 
        n_iterations=2
    )

    if g.is_connected():
        df = pd.DataFrame(columns=['AvgDegree', 'Diam', 'AvgShortPath','Clustering Coffe','r','Q'])
        AD = np.mean(g.degree())
        CC = g.transitivity_avglocal_undirected() 
        Diam = g.diameter()
        AvgShortPath = g.average_path_length()
        r = g.assortativity_degree()
        Q = leiden_comm.modularity
        df.loc[len(df)] = [AD, Diam, AvgShortPath,CC, r,Q]
        print(df)
    else: 
        print("unconnected graph")
        df = pd.DataFrame(columns=['AvgDegree','Clustering Coffe','r','Q'])
        AD = np.mean(g.degree())
        CC = g.transitivity_avglocal_undirected() 
        r = g.assortativity_degree()
        Q = leiden_comm.modularity
        df.loc[len(df)] = [AD,CC, r,Q]
        print(df)