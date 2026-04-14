import os
import pickle
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullFormatter

config = {
"font.family": ["Times New Roman", "SimHei"],
"font.size": 20,
"font.serif": ['SimHei'], # 中文黑体
}
plt.rcParams.update(config)
plt.rcParams["axes.unicode_minus"] = False  # 用 ASCII 减号，避免负号也变成方块

import powerlaw

from utils.palette import MAIN_METHOD_COLOR, PALETTE2
from utils.common import load_g

DATA_PATHS = [
    # "graphs/real/bio/foodweb-baywet",
    # "graphs/real/bio/maayan-foodweb",
    # "graphs/real/bio/arenas-meta",
    # "graphs/real/bio/maayan-vidal",
    # "graphs/real/social/petster-hamster",
    # "graphs/real/social/ego-twitter",
    # "graphs/real/social/loc-brightkite",
    # "graphs/real/social/loc-gowalla",
    # "graphs/real/information/web-EPA",
    # "graphs/real/information/subelj_jdk_jdk",
    # "graphs/real/information/linux",
    # "graphs/real/information/p2p-Gnutella31",
    # "graphs/real/tech/eu-powergrid",
    # "graphs/real/tech/gridkit-eupowergrid",
    # "graphs/real/tech/gridkit-north_america",
    # "graphs/real/tech/internet-topology",
    "graphs/example/econ-wm1",
    "graphs/example/eu-powergrid",
    "graphs/example/moreno_propro",
    "graphs/example/openflights-airport",
    "graphs/example/ego-facebook",
    "graphs/example/subelj_jdk_jdk",
    "graphs/example/dblp-cite",
    "graphs/example/linux",
    "graphs/example/slashdot",

]

NROWS = 3
NCOLS = 3


def _resolve_graph_path(base_path: str) -> Optional[str]:
    candidates = [base_path, f"{base_path}.pkl"]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def plot_powerlaw_grid(data_paths=DATA_PATHS, save_path="powerlaw_3x3.png"):
    nrows, ncols = NROWS, NCOLS
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows))

    empirical_color = MAIN_METHOD_COLOR
    fit_color = PALETTE2[0]

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            idx = r * ncols + c
            if idx >= len(data_paths):
                ax.axis("off")
                continue

            raw_path = data_paths[idx]
            graph_path = _resolve_graph_path(raw_path)
            graph_name = raw_path.split("/")[-1]
            ax.set_title(graph_name, fontsize=20, fontweight="medium")

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
                fr"$\lambda$" + f"={alpha:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=16,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
            )
            ax.grid(alpha=0.25)

            # powerlaw.plot_pdf 会设对数轴 + 默认 LogLocator(subs 含 2,5 等)，窄区间时主刻度过密；
            # 仅改 formatter 不够（仍会有 2×10^n 主刻度）。主刻度只保留 10^n，并去掉次刻度文字。
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
            ax.xaxis.set_major_formatter(LogFormatterMathtext())
            ax.xaxis.set_minor_formatter(NullFormatter())

            ax.set_ylabel("P(k)", fontsize=18)
            ax.set_xlabel("k", fontsize=18)

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
    plot_powerlaw_grid(save_path="visualizations/powerlaw_3x3.png")