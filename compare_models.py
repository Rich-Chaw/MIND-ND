#!/usr/bin/env python3
"""
Comprehensive comparison of dismantling methods including MIND checkpoints
Creates box plots for AUC and robustness metrics on BA, WS, and SBM graph types
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import igraph as ig
import os
import time
import warnings
from baseline import METHODS, evaluate_sol, ensure_attribute
from utils.graph_models import SBM, WS

warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    'n_graphs': 30,
    'nrange': "50_100",
    'max_test_graphs': 30,
    'baseline_methods': ["Degree", "Spectral", "Betweenness", "CI"],
    'mind_checkpoints': [
        'saved/mind.ckpt',
    ]
}
mind_dir = "saved/finetune_prior_20260108_214004"
max_step = 10000
gap = 200
# Get checkpoint files and sort by step number in descending order
checkpoint_files = [p for p in os.listdir(mind_dir)
                   if p.split('.')[0].isdigit() and int(p.split('.')[0]) < max_step and (int(p.split('.')[0])+1)%gap == 0]
checkpoint_files.sort(key=lambda x: int(x.split('.')[0]))
mind_checkpoints = [os.path.join(mind_dir, p) for p in checkpoint_files]
CONFIG["mind_checkpoints"].extend(mind_checkpoints)
# CONFIG["mind_checkpoints"].append('saved/finetune_20251231_163953/warmup/warmup_best_step_130_auc_0.2715.ckpt')
N_METHODS = len(CONFIG["baseline_methods"]) + len(CONFIG["mind_checkpoints"])

def dismantling_mind(g: ig.Graph, ckpt_pth: str, step_ratio: float = 0.0):
    """MIND dismantling method wrapper"""
    try:
        from networks.dismantle import SACPolicy
        import torch
        from utils.validate import validate_one_graph
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        sac = SACPolicy(num_features=16, num_heads=4, num_mps=6).to(device)
        sac.load_state_dict(torch.load(ckpt_pth, map_location=device)['policy_state_dict'])
        
        if 'name' not in g.attributes():
            g['name'] = f"test_graph_{g.vcount()}_{g.ecount()}"
        
        auc, robustness, _, removals = validate_one_graph(
            g, policy=sac, log_removals=True, step_ratio=step_ratio
        )
        return removals
        
    except Exception as e:
        print(f"Error with MIND checkpoint {os.path.basename(ckpt_pth)}: {e}")
        return None

def generate_test_graphs(n_graphs=50, nrange="100_150"):
    """Generate test graphs with parameter diversity"""
    print(f"Generating {n_graphs} graphs of each type with {nrange} nodes...")
    
    graphs = {'BA': [], 'WS': [], 'SBM': []}
    np.random.seed(42)
    n_min = int(nrange.split("_")[0])
    n_max = int(nrange.split("_")[1])

    # # BA graphs
    # i = 0
    # while i < n_graphs:
    #     n_nodes = np.random.randint(n_min,n_max+1)
    #     m = np.random.randint(1, 7)
    #     g = ig.Graph.Barabasi(n=n_nodes, m=m, directed=False)
    #     if g.is_connected() and g.vcount() >= n_min:
    #         ensure_attribute(g)
    #         g['name'] = f'BA_{i}'
    #         graphs['BA'].append(g)
    #         i +=1
    
    # # WS graphs
    # i = 0
    # while i < n_graphs:
    #     n_nodes = np.random.randint(n_min,n_max+1)
    #     k = np.random.choice([4, 6, 8, 10])
    #     p = np.random.uniform(0.05, 0.4)
    #     g = WS(n_nodes, k, p)
    #     if g.is_connected() and g.vcount() >= n_min:
    #         ensure_attribute(g)
    #         g['name'] = f'WS_{i}'
    #         graphs['WS'].append(g)
    #         i += 1
    
    # SBM graphs
    i = 0
    while i < n_graphs:
        n_nodes = np.random.randint(n_min,n_max+1)
        g = SBM(n_nodes, p_in=0.15, p_out=0.0075, num_blocks=2)
        # ratio = 5 * np.random.uniform(1, 6) 
        # p_out = max(0.01, np.random.rand() / 10)
        # p_in = min(ratio * p_out, 0.5)
        # num_blocks = np.random.randint(2, 5)
        # g = SBM(n_nodes, p_in, p_out, num_blocks)
        if g.is_connected() and g.vcount() >= n_min:
            ensure_attribute(g)
            g['name'] = f'SBM_{i}'
            graphs['SBM'].append(g)
            i += 1

    # Report statistics
    for graph_type, graph_list in graphs.items():
        if graph_list:
            sizes = [g.vcount() for g in graph_list]
            edges = [g.ecount() for g in graph_list]
            print(f"{graph_type}: {len(graph_list)} graphs, "
                  f"nodes: {np.mean(sizes):.1f}±{np.std(sizes):.1f}, "
                  f"edges: {np.mean(edges):.1f}±{np.std(edges):.1f}")
    
    return graphs

def test_method_on_graph(graph, method_name, method_func, method_type):
    """Test a single method on a single graph"""
    try:
        start_time = time.time()
        removals = method_func(graph)
        runtime = time.time() - start_time
        
        if removals is not None:
            auc, robustness = evaluate_sol(graph, removals)
            return {
                'method': method_name,
                'method_type': method_type,
                'auc': auc,
                'robustness': robustness,
                'runtime': runtime,
                'success': True
            }
        else:
            return {
                'method': method_name,
                'method_type': method_type,
                'auc': np.nan,
                'robustness': np.nan,
                'runtime': np.nan,
                'success': False
            }
    except Exception as e:
        print(f"    {method_name}: ERROR - {str(e)[:50]}...")
        return {
            'method': method_name,
            'method_type': method_type,
            'auc': np.nan,
            'robustness': np.nan,
            'runtime': np.nan,
            'success': False
        }
def run_comprehensive_comparison(graphs, baseline_methods, mind_checkpoints, max_graphs=30):
    """Run comparison with both baseline and MIND methods"""
    results = []
    
    for graph_type, graph_list in graphs.items():
        print(f"\n{'='*60}")
        print(f"Testing {graph_type} graphs...")
        
        test_graphs = graph_list[:max_graphs]
        
        for graph_idx, graph in enumerate(test_graphs):
            print(f"  Graph {graph_idx+1}/{len(test_graphs)} "
                  f"({graph.vcount()} nodes, {graph.ecount()} edges)")
            
            # Test baseline methods
            for method_name, method_func in baseline_methods.items():
                result = test_method_on_graph(graph, method_name, method_func, 'Baseline')
                result.update({
                    'graph_type': graph_type,
                    'graph_idx': graph_idx,
                    'nodes': graph.vcount(),
                    'edges': graph.ecount()
                })
                results.append(result)
                
                if result['success']:
                    print(f"    {method_name}: AUC={result['auc']:.3f}, R={result['robustness']:.3f}")
            
            # Test MIND checkpoints
            for ckpt_path in mind_checkpoints:
                if not os.path.exists(ckpt_path):
                    continue
                    
                # ckpt_name = ckpt_path.split("/")[1].split('_')[-1] + os.path.basename(ckpt_path).replace('.ckpt', '')
                ckpt_name = os.path.basename(ckpt_path)
                mind_func = lambda g: dismantling_mind(g, ckpt_path)
                
                result = test_method_on_graph(graph, ckpt_name, mind_func, 'MIND')
                result.update({
                    'graph_type': graph_type,
                    'graph_idx': graph_idx,
                    'nodes': graph.vcount(),
                    'edges': graph.ecount()
                })
                results.append(result)
                
                if result['success']:
                    print(f"    {ckpt_name}: AUC={result['auc']:.3f}, R={result['robustness']:.3f}")
    
    return pd.DataFrame(results)

def create_boxplot(df, metric, output_dir):
    """Create box plots for a specific metric"""
    fig, axes = plt.subplots(3, 1, figsize=(N_METHODS, 6*3))
    
    for i, graph_type in enumerate(['BA', 'WS', 'SBM']):
        # data = df[df['graph_type'] == graph_type].dropna(subset=[metric])
        data = df[df['graph_type'] == graph_type]  # seaborn handles NaN values
        
        if len(data) > 0:
            sns.boxplot(data=data, x='method', y=metric, hue='method_type', ax=axes[i])
            axes[i].set_title(f'{graph_type} Graphs - {metric.upper()} Distribution', fontsize=14)
            axes[i].set_xlabel('Method', fontsize=12)
            axes[i].set_ylabel(metric.upper(), fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            
            # # Add success rate
            # total = len(df[df['graph_type'] == graph_type])
            # successful = len(data)
            # success_rate = successful / total * 100 if total > 0 else 0
            # axes[i].text(0.02, 0.98, f'Success Rate: {success_rate:.1f}%', 
            #             transform=axes[i].transAxes, verticalalignment='top',
            #             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        else:
            axes[i].text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                        transform=axes[i].transAxes, fontsize=16)
            axes[i].set_title(f'{graph_type} Graphs - {metric.upper()} (No Data)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric}_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.show()

def create_success_rate_plot(df, output_dir):
    """Create success rate analysis plot"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, graph_type in enumerate(['BA', 'WS', 'SBM']):
        data = df[df['graph_type'] == graph_type]
        
        if len(data) > 0:
            success_rates = []
            methods = []
            method_types = []
            
            for method in data['method'].unique():
                method_data = data[data['method'] == method]
                total = len(method_data)
                successful = method_data['success'].sum()
                success_rate = successful / total * 100 if total > 0 else 0
                
                success_rates.append(success_rate)
                methods.append(method)
                method_types.append(method_data['method_type'].iloc[0])
            
            success_df = pd.DataFrame({
                'method': methods,
                'success_rate': success_rates,
                'method_type': method_types
            })
            
            sns.barplot(data=success_df, x='method', y='success_rate', hue='method_type', ax=axes[i])
            axes[i].set_title(f'{graph_type} Graphs - Success Rate', fontsize=14)
            axes[i].set_xlabel('Method', fontsize=12)
            axes[i].set_ylabel('Success Rate (%)', fontsize=12)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim(0, 105)
        else:
            axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', 
                        transform=axes[i].transAxes, fontsize=16)
            axes[i].set_title(f'{graph_type} Graphs - Success Rate (No Data)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()
def create_comprehensive_plots(df, output_dir="comprehensive_results"):
    """Create comprehensive comparison plots"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create plots
    create_boxplot(df, 'auc', output_dir)
    # create_boxplot(df, 'robustness', output_dir)
    # create_success_rate_plot(df, output_dir)
    
    # Statistical summary
    summary_stats = df.groupby(['graph_type', 'method', 'method_type']).agg({
        'auc': ['mean', 'std', 'count'],
        'robustness': ['mean', 'std'],
        'runtime': ['mean', 'std'],
        'success': 'sum'
    }).round(4)
    
    print("\n" + "="*100)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*100)
    print(summary_stats)
    
    # Save results
    summary_stats.to_csv(os.path.join(output_dir, 'detailed_summary.csv'))
    df.to_csv(os.path.join(output_dir, 'all_results.csv'), index=False)
    
    return summary_stats
def print_best_performers(df):
    """Print best performing methods for each graph type"""
    for graph_type in ['BA', 'WS', 'SBM']:
        print(f"\n{graph_type} Graphs:")
        type_data = df[df['graph_type'] == graph_type]
        
        if len(type_data) > 0:
            # Best AUC
            auc_data = type_data.dropna(subset=['auc'])
            if len(auc_data) > 0:
                best_auc = auc_data.groupby('method')['auc'].mean().sort_values()
                print(f"  Best AUC: {best_auc.index[0]} ({best_auc.iloc[0]:.4f})")
            
            # Best Robustness
            rob_data = type_data.dropna(subset=['robustness'])
            if len(rob_data) > 0:
                best_rob = rob_data.groupby('method')['robustness'].mean().sort_values()
                print(f"  Best Robustness: {best_rob.index[0]} ({best_rob.iloc[0]:.4f})")
            
            # Fastest
            runtime_data = type_data.dropna(subset=['runtime'])
            if len(runtime_data) > 0:
                fastest = runtime_data.groupby('method')['runtime'].mean().sort_values()
                print(f"  Fastest: {fastest.index[0]} ({fastest.iloc[0]:.4f}s)")

def main():
    """Main execution function"""
    print("Comprehensive Dismantling Methods Comparison")
    print("="*60)
    
    # Get baseline methods
    baseline_methods = {name: METHODS[name] for name in CONFIG['baseline_methods']}
    
    # Check available checkpoints
    available_checkpoints = [ckpt for ckpt in CONFIG['mind_checkpoints'] if os.path.exists(ckpt)]
    print(f"Found {len(available_checkpoints)} MIND checkpoints:")
    for ckpt in available_checkpoints:
        print(f"  - {ckpt}")
    
    # Generate test graphs
    graphs = generate_test_graphs(CONFIG['n_graphs'], CONFIG['nrange'])
    
    # Run comparison
    results_df = run_comprehensive_comparison(
        graphs, baseline_methods, available_checkpoints, CONFIG['max_test_graphs']
    )
    
    if len(results_df) > 0:
        # Create analysis
        timestamp = int(time.time())
        output_dir = f"comprehensive_results_{timestamp}"
        
        create_comprehensive_plots(results_df, output_dir)
        print_best_performers(results_df)
        
        print(f"\nAll results saved to: {output_dir}")
        print(f"Total experiments conducted: {len(results_df)}")
    else:
        print("No results generated. Check for errors.")

if __name__ == "__main__":
    main()