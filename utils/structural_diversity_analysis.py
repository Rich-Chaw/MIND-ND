#!/usr/bin/env python3
"""
Structural Diversity Analysis for MIND-ND Synthetic Graphs
Visualizes Modularity vs Assortativity by graph type (ER, LPA, Copy)
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import igraph as ig
from collections import Counter
import warnings
import powerlaw
warnings.filterwarnings('ignore')
from utils.palette import MAIN_METHOD_COLOR, _OTHER_METHOD_PALETTE

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

def load_graphs(data_dirs):
    """Load and convert graphs from pkl files"""
    graphs = []
    for source, data_dirs in data_dirs.items():
        for data_dir in data_dirs:
            for file in os.listdir(data_dir):
                if file.endswith('.pkl'):
                    with open(os.path.join(data_dir,file),'rb') as f:
                        graph = pickle.load(f)
                        graph["source"] = source
                        graphs.append(graph)
    return graphs

def statistics(graph):
    leiden_comm = graph.community_leiden(
        objective_function="modularity", 
        weights=None, 
        resolution_parameter=1.0, 
        n_iterations=2
    )

    if graph.is_connected():
        df = pd.DataFrame(columns=['Num_nodes','Num_edges','AvgDegree', 'Diam', 'AvgShortPath','Clustering','r','Q'])
        N = graph.vcount()
        E = graph.ecount()
        AD = np.mean(graph.degree())
        CC = graph.transitivity_avglocal_undirected() 
        Diam = graph.diameter()
        AvgShortPath = graph.average_path_length()
        r = graph.assortativity_degree()
        Q = leiden_comm.modularity
        df.loc[len(df)] = [N,E,AD, Diam, AvgShortPath,CC, r,Q]
        return df
        # print(df)
    else: 
        print("unconnected graph")
        df = pd.DataFrame(columns=['Num_nodes','Num_edges','AvgDegree','Clustering','r','Q'])
        N = graph.vcount()
        E = graph.ecount()
        AD = np.mean(graph.degree())
        CC = graph.transitivity_avglocal_undirected() 
        r = graph.assortativity_degree()
        Q = leiden_comm.modularity
        df.loc[len(df)] = [N,E,AD,CC, r,Q]
        return df

def analyze_powerlaw(graph, info=False, visualize=False):
    degrees = graph.degree()
    
    fit = powerlaw.Fit(degrees, discrete=True)
    
    # Calculate the Power-law Coefficient (alpha)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    
    # Significance Testing: Compare power_law vs exponential distribution
    # R is the loglikelihood ratio. Positive R favors the first distribution.
    R, p_value = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    if info:
        print(f"--- Power-law Analysis ---")
        print(f"Alpha (Coefficient): {alpha:.4f}")
        print(f"xmin (Threshold): {xmin}")
        print(f"Loglikelihood Ratio (R): {R:.4f}")
        print(f"p-value: {p_value:.4f}")
    
        if R > 0 and p_value < 0.05:
            print("Result: Power-law is significantly more likely than Exponential.")
        else:
            print("Result: Power-law distribution is NOT statistically significant.")

    # 5. Visualization
    if visualize:
        plt.figure() 
        empirical_color = MAIN_METHOD_COLOR
        fit_color = _OTHER_METHOD_PALETTE[0]
        fig = fit.plot_pdf(color=empirical_color, linewidth=2, label='Empirical Data')
        fit.power_law.plot_pdf(color=fit_color, linestyle='--', ax=fig, label='Power-law Fit')
        plt.xlabel('Degree (k)')
        plt.ylabel('P(k)')
        plt.legend()
        plt.show()

    return alpha, xmin, R, p_value

def calculate_properties(graphs):
    """Calculate modularity, assortativity and Clustering for all graphs; return a DataFrame."""
    rows = []

    for i, item in enumerate(graphs):
        if i % 1000 == 0:
            print(f"Processing {i+1}/{len(graphs)}")

        try:
            if isinstance(item, tuple):
                graph, topology = item
            else:
                graph = item
                topology = graph["config"]["topology"] if "config" in graph.attributes() else "unknown"

            source = graph["source"] if "source" in graph.attributes() else "unknown"

            if graph.vcount() < 10:
                continue

            stat_df = statistics(graph)
            if stat_df is None or stat_df.empty:
                continue

            stat_row = stat_df.iloc[0]
            rows.append({
                "Q": stat_row["Q"],
                "r": stat_row["r"],
                "Clustering": stat_row["Clustering"],
                "label": topology,
                "source": source,
            })

        except Exception as e:
            print(e)
            continue

    return pd.DataFrame(rows)

def create_scatter_plot(q_values, r_values, labels, save_path=None):
    """Create scatter plot with topology labels"""
    print(f"\nCreating scatter plot with {len(q_values)} points...")

    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    color_map = dict(zip(unique_labels, colors))
    
    plt.figure(figsize=(10, 6))
    
    for label in unique_labels:
        mask = labels == label
        plt.scatter(q_values[mask], r_values[mask], 
                   color=color_map[label], alpha=0.6, s=20, edgecolors='black',linewidth=0.3, label=label)
    
    plt.xlabel('Modularity (Q)')
    plt.ylabel('Assortativity (r)')
    plt.title('Structural Diversity: Modularity vs Assortativity by Graph Type')
    plt.legend(title='Graph Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add stats
    stats = f"n={len(q_values)} | Q∈[{q_values.min():.2f},{q_values.max():.2f}] | r∈[{r_values.min():.2f},{r_values.max():.2f}]"
    plt.figtext(0.02, 0.02, stats, fontsize=9, style='italic')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
    plt.show()

def plot_graph_metrics_boxplot(df, save_path=None):
    # df 需要包含: label, r, Q, Clustering
    metrics = [
        ("r", "Assortativity"),
        ("Q", "Modularity"),
        ("Clustering", "Clustering Coefficient"),
    ]

    groups = sorted(df["label"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=False)

    for ax, (col, title) in zip(axes, metrics):
        data = [df.loc[df["label"] == graph, col].dropna().values for graph in groups]
        ax.boxplot(data, labels=groups, showfliers=False)
        ax.set_title(title)
        ax.set_xlabel("Graph Group")
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)

    fig.suptitle("Graph Assortativity, Modularity and Clustering", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Scatter plot saved to {save_path}")
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_pairwise_similarity(df, save_path=None, metric_cols=("Clustering", "Q", "r"), source_col="source"):
    """
    Plot a compact 2x3 summary of structural similarity for synthetic and real graphs.

    Top row:
        density of Clustering, Q, r
    Bottom row:
        scatter plots of (Clustering, Q), (Clustering, r), (Q, r)

    Args:
        df: pandas DataFrame containing metric columns and a source column
        save_path: optional path to save the figure
        metric_cols: must contain exactly ("Clustering", "Q", "r") by default
        source_col: column indicating graph origin, e.g. "synthetic" / "generated" / "real"

    Returns:
        fig, axes
    """
    missing = [c for c in list(metric_cols) + [source_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(metric_cols) != 3:
        raise ValueError("plot_pairwise_similarity expects exactly 3 metric columns.")

    plot_df = df[list(metric_cols) + [source_col]].dropna().copy()
    sources = list(plot_df[source_col].unique())

    def _source_color(src):
        src_lower = str(src).strip().lower()
        if src_lower == "real":
            return "#d62728"
        if src_lower in {"synthetic", "generated", "gen", "fake"}:
            return "#7f7f7f"
        return "#7f7f7f"

    color_map = {src: _source_color(src) for src in sources}
    display_names = {
        "Clustering": "Clustering Coefficient",
        "Q": "Modularity (Q)",
        "r": "Assortativity (r)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Top row: density-style normalized histograms
    for ax, metric in zip(axes[0], metric_cols):
        for src in sources:
            vals = plot_df.loc[plot_df[source_col] == src, metric].values
            if len(vals) == 0:
                continue
            ax.hist(
                vals,
                bins=20,
                density=True,
                alpha=0.45,
                color=color_map[src],
                edgecolor="black",
                linewidth=0.3,
                label=str(src),
            )
        ax.set_title(f"{display_names.get(metric, metric)} Density")
        ax.set_xlabel(display_names.get(metric, metric))
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)

    # Bottom row: pairwise scatter plots
    pair_specs = [
        ("Clustering", "Q"),
        ("Clustering", "r"),
        ("Q", "r"),
    ]
    for ax, (x_col, y_col) in zip(axes[1], pair_specs):
        for src in sources:
            sub = plot_df.loc[plot_df[source_col] == src]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub[x_col].values,
                sub[y_col].values,
                s=28,
                alpha=0.65,
                color=color_map[src],
                edgecolors="black",
                linewidths=0.2,
                label=str(src),
            )
        ax.set_title(f"{display_names.get(x_col, x_col)} vs {display_names.get(y_col, y_col)}")
        ax.set_xlabel(display_names.get(x_col, x_col))
        ax.set_ylabel(display_names.get(y_col, y_col))
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)

    fig.suptitle("Pairwise Structural Similarity of Synthetic and Real Graphs", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Pairwise similarity plot saved to {save_path}")

    plt.show()
    return fig, axes


def compute_mahalanobis_similarity(
    df,
    metric_cols=("Q", "r", "Clustering"),
    source_col="source",
    reg=1e-6,
):
    """
    Compute a Gaussian-style similarity between synthetic and real graph distributions
    using Mahalanobis distance between the two group means with pooled covariance.

    This is suitable when both synthetic and real graphs contain multiple samples.

    Args:
        df: DataFrame containing both synthetic and real graphs
        metric_cols: metric columns used jointly
        source_col: column indicating graph origin, e.g. "synthetic" / "generated" / "real"
        reg: diagonal regularization added to covariance for numerical stability

    Returns:
        result: dict with keys
            - distance: Mahalanobis distance between mean vectors
            - similarity: exp(-0.5 * distance^2), in (0, 1]
            - mean_synthetic
            - mean_real
            - covariance
            - n_synthetic
            - n_real
    """
    missing = [c for c in list(metric_cols) + [source_col] if c not in df.columns]
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    work_df = df[list(metric_cols) + [source_col]].dropna().copy()
    source_series = work_df[source_col].astype(str).str.strip().str.lower()

    gen_mask = source_series.isin(["synthetic", "generated", "gen", "fake"])
    real_mask = source_series.eq("real")

    Xg = work_df.loc[gen_mask, list(metric_cols)].to_numpy(dtype=float)
    Xr = work_df.loc[real_mask, list(metric_cols)].to_numpy(dtype=float)

    if len(Xg) < 2 or len(Xr) < 2:
        raise ValueError("df must contain at least 2 valid synthetic/generated samples and 2 valid real samples.")

    mu_g = Xg.mean(axis=0)
    mu_r = Xr.mean(axis=0)

    cov_g = np.cov(Xg, rowvar=False)
    cov_r = np.cov(Xr, rowvar=False)

    # Pooled covariance
    pooled_cov = ((len(Xg) - 1) * cov_g + (len(Xr) - 1) * cov_r) / (len(Xg) + len(Xr) - 2)
    pooled_cov = pooled_cov + reg * np.eye(len(metric_cols))

    inv_cov = np.linalg.pinv(pooled_cov)
    delta = mu_g - mu_r
    distance = float(np.sqrt(delta.T @ inv_cov @ delta))

    # Map distance to a bounded similarity score
    similarity = float(np.exp(-0.5 * distance ** 2))

    return {
        "distance": distance,
        "similarity": similarity,
        "mean_synthetic": dict(zip(metric_cols, mu_g)),
        "mean_real": dict(zip(metric_cols, mu_r)),
        "covariance": pooled_cov,
        "n_synthetic": int(len(Xg)),
        "n_real": int(len(Xr)),
    }


def compute_joint_mmd(
    df,
    metric_cols=("Q", "r", "Clustering"),
    source_col="source",
    gamma=None,
    reg_eps=1e-12,
):
    """
    Compute joint 3D MMD between synthetic and real graph distributions.

    Uses an RBF kernel on jointly standardized features.
    By default, bandwidth is chosen with the median heuristic.

    Args:
        df: DataFrame containing both synthetic and real graphs
        metric_cols: usually ("Q", "r", "Clustering")
        source_col: column indicating graph origin, e.g. "synthetic" / "generated" / "real"
        gamma: RBF kernel parameter. If None, use median heuristic:
               gamma = 1 / (2 * median_dist^2)
        reg_eps: tiny value for numerical stability

    Returns:
        result: dict with keys
            - mmd2_unbiased: unbiased estimate of MMD^2
            - mmd_unbiased: sqrt(max(MMD^2, 0))
            - gamma
            - sigma
            - n_synthetic
            - n_real
    """
    missing = [c for c in list(metric_cols) + [source_col] if c not in df.columns]
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    work_df = df[list(metric_cols) + [source_col]].dropna().copy()
    source_series = work_df[source_col].astype(str).str.strip().str.lower()

    gen_mask = source_series.isin(["synthetic", "generated", "gen", "fake"])
    real_mask = source_series.eq("real")

    X = work_df.loc[gen_mask, list(metric_cols)].to_numpy(dtype=float)
    Y = work_df.loc[real_mask, list(metric_cols)].to_numpy(dtype=float)

    if len(X) < 2 or len(Y) < 2:
        raise ValueError("df must contain at least 2 valid synthetic/generated samples and 2 valid real samples.")

    # Joint standardization to prevent one metric dominating by scale
    Z = np.vstack([X, Y])
    mean = Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, keepdims=True)
    std[std < reg_eps] = 1.0

    Xs = (X - mean) / std
    Ys = (Y - mean) / std

    def sq_dists(A, B):
        a2 = np.sum(A * A, axis=1, keepdims=True)
        b2 = np.sum(B * B, axis=1, keepdims=True).T
        d2 = a2 + b2 - 2.0 * A @ B.T
        return np.maximum(d2, 0.0)

    # Median heuristic for bandwidth
    if gamma is None:
        ZZ = np.vstack([Xs, Ys])
        D2 = sq_dists(ZZ, ZZ)
        tri = D2[np.triu_indices_from(D2, k=1)]
        tri = tri[tri > reg_eps]
        if len(tri) == 0:
            sigma2 = 1.0
        else:
            sigma2 = np.median(tri)
        gamma = 1.0 / (2.0 * sigma2 + reg_eps)

    Kxx = np.exp(-gamma * sq_dists(Xs, Xs))
    Kyy = np.exp(-gamma * sq_dists(Ys, Ys))
    Kxy = np.exp(-gamma * sq_dists(Xs, Ys))

    n = len(Xs)
    m = len(Ys)

    # Unbiased estimator
    sum_Kxx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    sum_Kyy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sum_Kxy = Kxy.mean()

    mmd2 = float(sum_Kxx + sum_Kyy - 2.0 * sum_Kxy)
    mmd = float(np.sqrt(max(mmd2, 0.0)))
    sigma = float(np.sqrt(1.0 / (2.0 * gamma)))

    return {
        "mmd2_unbiased": mmd2,
        "mmd_unbiased": mmd,
        "gamma": float(gamma),
        "sigma": sigma,
        "n_synthetic": int(n),
        "n_real": int(m),
    }

def create_heatmap(q_values, r_values, labels, save_path = None):
    """
    Create heatmap showing distribution density by topology
    """
    print("\nCreating topology-specific heatmaps...")
    
    # Convert to numpy arrays
    q_values = np.array(q_values)
    r_values = np.array(r_values)
    labels = np.array(labels)
    
    unique_labels = np.unique(labels)
    n_topologies = len(unique_labels)
    
    # Create subplots for each topology
    fig, axes = plt.subplots(1, n_topologies, figsize=(5*n_topologies, 4))
    if n_topologies == 1:
        axes = [axes]
    
    for i, topology in enumerate(unique_labels):
        mask = labels == topology
        q_topo = q_values[mask]
        r_topo = r_values[mask]
        
        # Create 2D histogram
        hist, q_edges, r_edges = np.histogram2d(q_topo, r_topo, bins=15)
        
        # Plot heatmap
        im = axes[i].imshow(hist.T, origin='lower', aspect='auto', cmap='viridis',
                            extent=[q_edges[0], q_edges[-1], r_edges[0], r_edges[-1]])
        
        axes[i].set_xlabel('Modularity (Q)')
        axes[i].set_ylabel('Assortativity (r)')
        axes[i].set_title(f'{topology} (n={np.sum(mask)})')
        axes[i].grid(True, alpha=0.3)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Count')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Heatmap saved to {save_path}")
    
def uniformity(q_values, r_values) -> float:
    """
    Quantify uniformity using 2D binning and coefficient of variation
    
    Returns:
        Uniformity coefficient of variation (lower is more uniform)
    """
    print("\nAnalyzing uniformity...")
    
    # Convert to numpy arrays
    q_values = np.array(q_values)
    r_values = np.array(r_values)
    
    # Create 2D histogram
    num_bins = 10
    hist, q_edges, r_edges = np.histogram2d(q_values, r_values, bins=num_bins)
    
    # Calculate uniformity metric (Coefficient of Variation)
    non_empty_bins = hist[hist > 0]
    if len(non_empty_bins) > 0:
        uniformity_cv = np.std(non_empty_bins) / np.mean(non_empty_bins)
    else:
        uniformity_cv = float('inf')
    
    print(f"Uniformity CV: {uniformity_cv:.4f} (lower is more uniform)")
    print(f"Non-empty bins: {len(non_empty_bins)}/{num_bins**2}")
    
    return uniformity_cv
    
def main():
    """Main analysis function"""
    data_dirs = {
        'synthetic': [
            'graphs/train/100_200_BA_1000'
        ],
        'real': [
            'graphs/real/bio'
        ]
    }

    # Load and analyze
    graphs = load_graphs(data_dirs)

    print(f"Loaded {len(graphs)} graphs")
    
    df = calculate_properties(graphs)
    plot_graph_metrics_boxplot(df, save_path='test_boxplot.png')
    plot_pairwise_similarity(df, save_path='test_pairwise_similarity.png')
    print(compute_mahalanobis_similarity(df))
    print(compute_joint_mmd(df))

    # # Visualize and summarize
    # create_scatter_plot(q_values, r_values, labels)
    # create_heatmap(q_values, r_values, labels)
    # uniformity_cv = uniformity(q_values, r_values)
    
    # print(f"Uniformity CV: {uniformity_cv:.4f}")
    # for label in np.unique(labels):
    #     mask = labels == label
    #     q_sub, r_sub = q_values[mask], r_values[mask]
    # q_values, r_values, labels = calculate_properties(graphs)
    
    # print(f"\n{label}: Q={q_sub.mean():.3f}±{q_sub.std():.3f}, r={r_sub.mean():.3f}±{r_sub.std():.3f}")

if __name__ == "__main__":
    main()