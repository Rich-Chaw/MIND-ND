import igraph as ig
import networkx as nx
import subprocess
import json
import tempfile
import pickle
import os

def finder(graph):
    """
    Use AAA-NetDQN to dismantle a graph
    
    Args:
        graph: igraph.Graph object
        
    Returns:
        tuple: (removed_nodes_list, score,MaxCCList)
    """
    # Point directly to the python executable in your TF conda env
    tf_python_path = "D:\\Anaconda3\\envs\\tf_py37\\python.exe"
    # import sys
    # tf_python_path = sys.executable
    
    
    # Create temporary pickle file for the graph, absolute path
    temp_graph_fd, temp_graph_file = tempfile.mkstemp(suffix='.pkl')
    temp_out_fd, temp_out_file = tempfile.mkstemp(suffix='.json')
    # Close file descriptors immediately to avoid Windows permission issues
    os.close(temp_graph_fd)
    os.close(temp_out_fd)

    # temp_graph_file = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)
    # temp_out_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
    
    try:
        # Save graph to temporary pickle file
        with open(temp_graph_file, 'wb') as f:
            pickle.dump(graph, f)
        
        # Path to the python_interface.py
        interface_path = os.path.join("O:\My_Codes\GD2026\AAA-NetDQN", "python_interface.py")
        
        # Run the interface
        result = subprocess.run(
            [tf_python_path, interface_path, "--graph_file", temp_graph_file, "--out", temp_out_file],
            capture_output=True, text=True, cwd=os.path.dirname(interface_path)
        )
        
        if result.returncode == 0:
            # Parse JSON output
            # output_data = json.loads(result.stdout.strip())
            with open(temp_out_file, 'r') as f:
                output_data = json.load(f)
            removed_nodes = output_data['removed_nodes']
            score = output_data['score']
            MaxCCList = output_data['MaxCCList']
            return removed_nodes, score, MaxCCList
        else:
            print(f"Error running finder dismantling: {result.stderr}")
            return [], 0.0, []
    except Exception as e:
        print(f"Error in finder: {e}")
        return [], 0.0, []

    finally:
        # Clean up temporary file
        if os.path.exists(temp_graph_file):
            os.remove(temp_graph_file)
        if os.path.exists(temp_out_file):
            os.remove(temp_out_file)
        

if __name__ == "__main__":
    # Create a simple test graph
    G = nx.barabasi_albert_graph(20, 2)
    
    # Convert to igraph for testing
    import igraph as ig
    # Convert networkx to igraph
    edges = list(G.edges())
    ig_graph = ig.Graph(edges=edges, directed=False)
    
    import pickle
    with open("../graphs/real/FINDER/Crime.pkl",'rb') as f:
        ig_graph = pickle.load(f)
    
    print("Testing finder with Barabási-Albert graph (20 nodes, m=2)")
    removed_nodes, score = finder(ig_graph)
    
    print(f"Removed nodes: {removed_nodes}")
    print(f"Score: {score}")
    