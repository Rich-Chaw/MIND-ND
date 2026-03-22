"""
从 xlsx 读取各方法在各数据集上的指标，按分组画图：
每组一张图，含 overall 条形图 + 各数据集上的相对指标散点图，以指定主列为基准。
风格与颜色参考 visualize_dismantling.visualize_multiple_curve（tab20 配色）。
"""

import os
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro


def load_metric_xlsx(path, sheet_name=0):
    """
    从 xlsx 读取为 DataFrame。
    首行为标题：第一列为分组(Group)，第二列为数据集名(Dataset)，其余为方法名(指标列)。
    空单元格读作 NaN。
    sheet_name: 工作表名或索引，默认 0（第一张表）。
    """
    df = pd.read_excel(path, header=0, sheet_name=sheet_name)
    cols = list(df.columns)
    if len(cols) < 2:
        raise ValueError("xlsx 至少需要两列：分组、数据集名")
    # 第一列：分组，可能无列名或合并单元格；第二列：数据集名
    c0, c1 = cols[0], cols[1]
    if str(c0).strip() != "Group":
        df = df.rename(columns={c0: "Group"})
    if str(c1).strip() != "Dataset":
        df = df.rename(columns={c1: "Dataset"})
    # 分组列可能因合并单元格出现 NaN，向前填充
    df["Group"] = df["Group"].ffill()
    # 删除无数据集名的行
    df = df.dropna(subset=["Dataset"], how="all")
    df = df.dropna(subset=["Group"], how="all")
    # 若有 random 列则删掉
    random_col = next((c for c in df.columns if str(c).strip().lower() == "random"), None)
    if random_col is not None:
        df = df.drop(columns=[random_col])
    return df


def get_method_columns(df):
    """返回方法列（除 Group、Dataset 外的列，用于数值指标）。"""
    skip = {"Group", "Dataset"}
    return [c for c in df.columns if c not in skip]


def get_main_column(df, method_cols, main_col):
    """在方法列中找到与 main_col 匹配的列（不区分大小写）。"""
    target = str(main_col).strip().lower()
    for c in method_cols:
        if str(c).strip().lower() == target:
            return c
    return None


def compute_relative_to_main(df, method_cols, main_col, baseline=100):
    """
    按行计算相对值：以 main_col 为 baseline，其他方法 = value / main_value * baseline。
    返回 (df_with_relative, method_cols_ordered)，method_cols_ordered 中 main_col 在前。
    """
    if main_col not in method_cols or main_col not in df.columns:
        raise ValueError(f"未找到 {main_col}，无法归一化")
    out = df[["Group", "Dataset"]].copy()
    others = [c for c in method_cols if c != main_col]
    ordered = [main_col] + others

    for m in method_cols:
        rel = np.full(len(df), np.nan)
        main_vals = df[main_col].values
        vals = df[m].values
        valid = np.isfinite(main_vals) & (main_vals != 0) & np.isfinite(vals)
        rel[valid] = vals[valid] / main_vals[valid] * baseline
        out[m] = rel
    return out, ordered


from utils.palette import _OTHER_METHOD_PALETTE, MAIN_METHOD_COLOR


def _method_colors(method_cols, main_col=None):
    """配色：主列固定红色，其余从 _OTHER_METHOD_PALETTE 按序取色，不重复且与红色区分明显。"""
    out = {}
    idx = 0
    for m in method_cols:
        if main_col is not None and m == main_col:
            out[m] = MAIN_METHOD_COLOR
        else:
            out[m] = _OTHER_METHOD_PALETTE[idx % len(_OTHER_METHOD_PALETTE)]
            idx += 1
    return out


def plot_one_group(
    df_rel,
    group_name,
    method_cols_ordered,
    main_col,
    baseline=100,
    save_path=None,
    xlabel="relative AUC",
):
    """
    画一组的一个图：上为 overall 条形图，下为各数据集上的相对值散点图。
    """
    color_map = _method_colors(method_cols_ordered, main_col=main_col)
    fig, (ax_bar, ax_scatter) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1, 2]})

    # 1) Overall 条形图：组内各数据集上各方法的相对值取平均
    means = df_rel[method_cols_ordered].mean(skipna=True)
    means = means.reindex(method_cols_ordered).fillna(0)
    x_pos = np.arange(len(method_cols_ordered))
    bars = ax_bar.bar(x_pos, means.values, color=[color_map[m] for m in method_cols_ordered], alpha=0.85, edgecolor="gray", linewidth=0.5)
    ax_bar.axhline(y=baseline, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([str(m) for m in method_cols_ordered], rotation=45, ha="right")
    ax_bar.set_ylabel(xlabel)
    ax_bar.set_title("overall")
    ax_bar.set_ylim(0, max(means.max() * 1.1, baseline * 1.2))

    # 2) 散点图：每个数据集一行，x 为相对值，按方法着色，主列用星形
    datasets = df_rel["Dataset"].astype(str).tolist()
    n_datasets = len(datasets)
    y_pos = np.arange(n_datasets)
    ax_scatter.axvline(x=baseline, color="gray", linestyle="--", linewidth=1, alpha=0.8, label=f"{xlabel}={baseline}")

    for j, ds in enumerate(datasets):
        row = df_rel.iloc[j]
        for m in method_cols_ordered:
            val = row.get(m, np.nan)
            if not np.isfinite(val):
                continue
            if m == main_col:
                ax_scatter.scatter(val, j, c=[color_map[m]], s=180, marker="*", zorder=3, edgecolors="darkred", linewidths=0.8)
            else:
                ax_scatter.scatter(val, j, c=[color_map[m]], s=50, alpha=0.85, edgecolors="gray", linewidths=0.3)

    ax_scatter.set_yticks(y_pos)
    ax_scatter.set_yticklabels(datasets, fontsize=9)
    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel("Dataset")
    ax_scatter.set_title(group_name)
    ax_scatter.set_ylim(-0.5, n_datasets - 0.5)

    # 图例：方法名 + 颜色
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[m], markersize=8, label=m)
        for m in method_cols_ordered
    ]
    ax_scatter.legend(handles=legend_handles, loc="upper right", fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
        plt.close()
    else:
        plt.show()
    return fig


def visualize_metric_xlsx(
    xlsx_path,
    out_dir=None,
    xlabel="relative AUC",
    sheet_name=0,
    main_col="mind",
    baseline=100.0,
):
    """
    主入口：读取 xlsx，按分组画图并可选保存。
    - xlsx_path: 表格路径（首行标题，第一列分组，第二列数据集，其余方法）
    - out_dir: 若提供则按组名保存图片到此目录
    - xlabel: 散点图 x 轴标签
    - sheet_name: 工作表名或索引
    - main_col: 作为基准的主列名
    - baseline: 主列对应的基准数值
    """
    df = load_metric_xlsx(xlsx_path, sheet_name=sheet_name)
    method_cols = get_method_columns(df)
    # 方法列转为数值，空单元格变为 NaN
    for c in method_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    main_col = get_main_column(df, method_cols, main_col)
    if main_col is None:
        raise ValueError(f"表中未找到列「{main_col}」，请确保存在该列（不区分大小写）")

    df_rel, method_cols_ordered = compute_relative_to_main(df, method_cols, main_col, baseline)
    groups = df_rel["Group"].dropna().unique().tolist()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for g in groups:
        sub = df_rel[df_rel["Group"] == g].copy()
        if sub.empty:
            continue
        save_path = None
        if out_dir:
            safe_name = str(g).replace(os.sep, "_").strip()
            save_path = os.path.join(out_dir, f"metric_{safe_name}.png")
        plot_one_group(sub, str(g), method_cols_ordered, main_col, baseline, save_path=save_path, xlabel=xlabel)
    return df, df_rel, groups


@dataclass
class Args:
    """可视化指标脚本的参数"""

    xlsx_path: str = "测试数据.xlsx"
    """xlsx 表格路径"""
    out_dir: str = "plots"
    """图片输出目录"""
    xlabel: str = "relative R"
    """散点图 x 轴标签"""
    sheet: Optional[str] = "Sheet2"
    """工作表名或索引，默认第一张表；若为数字字符串则按索引解析"""
    no_print: bool = False
    """不打印 DataFrame 预览"""
    main_col: str = "mind"
    """作为基准的主列名（不区分大小写），相对值以此列为基准"""
    baseline: float = 100.0
    """主列对应的基准数值，其他方法按比例换算为此基准"""


if __name__ == "__main__":
    args = tyro.cli(Args)
    sheet_name = 0 if args.sheet is None else (int(args.sheet) if args.sheet.isdigit() else args.sheet)
    df_raw, df_rel, groups = visualize_metric_xlsx(
        args.xlsx_path,
        out_dir=args.out_dir,
        xlabel=args.xlabel,
        sheet_name=sheet_name,
        main_col=args.main_col,
        baseline=args.baseline,
    )
    if not args.no_print:
        print("DataFrame 预览（原始）:")
        print(df_raw.head(10))
        print(f"\n相对值（{args.main_col}={args.baseline}）预览:")
        print(df_rel.head(10))
        print("\n分组:", groups)
