import os
import pickle
from typing import Optional

import matplotlib.pyplot as plt

config = {
"font.family": ["Times New Roman", "SimSun"],
"font.size": 14,
"font.serif": ['SimSun'], # 中文宋体
}
plt.rcParams.update(config)
plt.rcParams["axes.unicode_minus"] = False  # 用 ASCII 减号，避免负号也变成方块

import powerlaw

from utils.palette import MAIN_METHOD_COLOR, _OTHER_METHOD_PALETTE
from utils.common import load_g

DATA_PATHS = [
    "graphs/real/bio/foodweb-baywet",
    "graphs/real/bio/maayan-foodweb",
    # "graphs/real/bio/arenas-meta",
    "graphs/real/bio/maayan-vidal",
    "graphs/real/social/petster-hamster",
    "graphs/real/social/ego-twitter",
    # "graphs/real/social/loc-brightkite",
    "graphs/real/social/loc-gowalla",
    "graphs/real/information/web-EPA",
    "graphs/real/information/subelj_jdk_jdk",
    # "graphs/real/information/linux",
    "graphs/real/information/p2p-Gnutella31",
    "graphs/real/tech/eu-powergrid",
    # "graphs/real/tech/gridkit-eupowergrid",
    "graphs/real/tech/gridkit-north_america",
    "graphs/real/tech/internet-topology",
]

ROW_ORDER = ["bio", "social", "information", "tech"]


def _resolve_graph_path(base_path: str) -> Optional[str]:
    candidates = [base_path, f"{base_path}.pkl"]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def plot_powerlaw_grid(data_paths=DATA_PATHS, save_path="powerlaw_real_16.png"):
    grouped = {k: [] for k in ROW_ORDER}
    for p in data_paths:
        parts = p.replace("\\", "/").split("/")
        if len(parts) < 4:
            continue
        group_name = parts[-2]
        if group_name in grouped:
            grouped[group_name].append(p)

    nrows = len(ROW_ORDER)
    ncols = max(len(grouped[g]) for g in ROW_ORDER)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))

    if nrows == 1:
        axes = [axes]

    empirical_color = MAIN_METHOD_COLOR
    fit_color = _OTHER_METHOD_PALETTE[0]

    for r, group in enumerate(ROW_ORDER):
        row_paths = grouped[group]
        for c in range(ncols):
            ax = axes[r][c] if nrows > 1 else axes[c]
            if c >= len(row_paths):
                ax.axis("off")
                continue

            raw_path = row_paths[c]
            graph_path = _resolve_graph_path(raw_path)
            graph_name = raw_path.split("/")[-1]
            ax.set_title(graph_name, fontsize=14, fontweight="medium")

            if graph_path is None:
                ax.text(0.5, 0.5, "File not found", ha="center", va="center", fontsize=9)
                ax.axis("off")
                continue

            graph = load_g(graph_path, graph_name)
            degrees = graph.degree()
            fit = powerlaw.Fit(degrees, discrete=True, verbose=False)
            fit.plot_pdf(ax=ax, color=empirical_color, linewidth=2, label="Empirical")
            fit.power_law.plot_pdf(ax=ax, color=fit_color, linestyle="--", linewidth=2, label="Power-law Fit")

            alpha = fit.power_law.alpha
            xmin = fit.power_law.xmin
            ax.text(
                0.95,
                0.95,
                f"alpha={alpha:.2f}\nxmin={xmin:.0f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
            )
            ax.grid(alpha=0.25)

            if c == 0:
                ax.set_ylabel(f"{group}\nP(k)")
            else:
                ax.set_ylabel("P(k)")
            ax.set_xlabel("k")

    handles = [
        plt.Line2D([0], [0], color=empirical_color, lw=2, label="实际度分布"),
        plt.Line2D([0], [0], color=fit_color, lw=2, linestyle="--", label="拟合幂律度分布"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    # fig.suptitle("Power-law Distributions of 16 Real Networks", fontsize=16, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_powerlaw_grid(save_path="visualizations/powerlaw_real_16.png")