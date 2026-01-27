import igraph as ig
import networkx as nx
import subprocess
import json
import tempfile
import pickle
import os

def finder(graph:ig.Graph):
    """
    Use FINDER to dismantle a graph
    
    Args:
        graph: igraph.Graph object
        
    Returns:
        tuple: (removed_nodes_list, score, MaxCCList)
    """
    tf_python_path = "D:\\Anaconda3\\envs\\tf_py37\\python.exe"

    # Create temporary pickle file for the graph, absolute path
    temp_graph_fd, temp_graph_file = tempfile.mkstemp(suffix='.pkl')
    temp_out_fd, temp_out_file = tempfile.mkstemp(suffix='.json')
    os.close(temp_graph_fd)
    os.close(temp_out_fd)

    try:
        # Save graph to temporary pickle file
        with open(temp_graph_file, 'wb') as f:
            pickle.dump(graph, f)
        
        # Path to the python_interface.py
        interface_path = os.path.join("O:\My_Codes\GD2026\AAA-NetDQN", "python_interface.py")
        
        # Run the interface
        result = subprocess.run(
            [tf_python_path, interface_path, "--graph_file", temp_graph_file, "--out_file", temp_out_file],
            capture_output=True, text=True, cwd=os.path.dirname(interface_path)
        )
        
        if result.returncode == 0:
            with open(temp_out_file, 'r') as f:
                output_data = json.load(f)
            removals = output_data['removals']
            score = output_data['score']
            MaxCCList = output_data['MaxCCList']
            return removals, score, MaxCCList
        else:
            print(f"Error in finder returncode: {result.stderr}")
            return None, 0.0, None
    except Exception as e:
        print(f"Error in finder: {e}")
        return None, 0.0, None

    finally:
        # Clean up temporary file
        if os.path.exists(temp_graph_file):
            os.remove(temp_graph_file)
        if os.path.exists(temp_out_file):
            os.remove(temp_out_file)
        

if __name__ == "__main__":
    import pickle
    with open("../graphs/real/FINDER/Crime.pkl",'rb') as f:
        graph = pickle.load(f)
    
    removals, score, _ = finder(graph)
    
    import sys
    sys.path.append("..")
    from baseline import evaluate_sol
    auc, robustness = evaluate_sol(graph,removals)

    graph = graph.to_networkx()
    # Calculate robustness in forward order (same as EvaluateSol)
    # This removes nodes one by one and tracks the largest connected component
    graph_test = graph.copy()
    num_nodes = graph.number_of_nodes()
    total_max_num = 0.0
    max_wcc_sz_list_forward = []
    # Remove nodes in forward order (same as dismantling process)
    for node in removals:
        # Remove the node from the graph
        if node in graph_test:
            graph_test.remove_node(node)
        
        # Find the largest connected component after removal
        if graph_test.number_of_nodes() > 0:
            max_cc_size = max(len(c) for c in nx.connected_components(graph_test))
        else:
            max_cc_size = 0
        
        total_max_num += max_cc_size
        max_wcc_sz_list_forward.append(max_cc_size / num_nodes)
    robustness_forward = total_max_num / (num_nodes * num_nodes)
    
    print(f"Removed nodes: {removals}")
    print(f"Score: {score}")
    