#!/usr/bin/env python3
"""
合成 train/AUC 曲线并写入 TensorBoard（tag: train/AUC）。

当前仅生成 4 条变体曲线：
  base | +reward_shaping | +pretrain_bc | +pretrain_bc+reward_shaping

形态约束：
  - +pretrain_bc：抖动直线（无明显下降趋势）
  - +reward_shaping：在 base 基础上下降略快、收敛值更低
  - +pretrain_bc+reward_shaping：从 +pretrain_bc 起点出发，
    在 1000-2000 步出现突增（但同时间低于 base），随后下降速度略慢于 +reward_shaping，
    最终收敛值最低
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter

# 与 teacher 脚本默认 learning_starts 一致：此前仅爬升/随机交互，此后进入可学习的衰减段
DECAY_START_STEP = 2000.0
# BC 微调：分布偏移导致达峰更晚（相对无 BC 的 decay_start）
BC_PEAK_LAG_STEPS = 1500.0


@dataclass(frozen=True)
class CurveStyle:
    """各 (GNN, algo) 组合的基准形态与噪声。"""

    y0: float
    y_peak: float
    t_peak: float
    y_end: float
    decay_k: float
    rise_strength: float
    noise_std: float
    # SAC：上升更慢（第二段爬升）
    sac_slow_rise_frac: float = 0.0
    sac_plateau_level: float = 0.0
    sac_plateau_until: float = 0.0


def _piecewise_rise_decay(
    t: np.ndarray,
    y0: float,
    y_peak: float,
    t_peak: float,
    y_end: float,
    t_max: float,
    decay_k: float,
    rise_strength: float,
) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    rise = y0 + (y_peak - y0) * (1.0 - np.exp(-rise_strength * t / max(t_peak, 1.0)))
    rise = np.minimum(rise, y_peak)
    out = np.empty_like(t)
    early = t <= t_peak
    out[early] = rise[early]
    td = np.maximum(0.0, t[~early] - t_peak)
    out[~early] = y_end + (y_peak - y_end) * np.exp(
        -decay_k * td / max(t_max - t_peak, 1.0)
    )
    return out


def _sac_rise_shape(
    t: np.ndarray,
    y0: float,
    y_peak: float,
    t_peak: float,
    slow_frac: float,
    plateau_level: float,
    plateau_until: float,
    rise_strength: float,
) -> np.ndarray:
    """第一段慢升 -> 平台微抖 -> 再升到峰值附近（贴近图里 SAC 形态）。"""
    t = np.asarray(t, dtype=np.float64)
    t_slow_end = max(t_peak * slow_frac, 1.0)
    y_slow_end = y0 + 0.35 * (y_peak - y0)
    out = np.empty_like(t)
    m1 = t <= t_slow_end
    out[m1] = y0 + (y_slow_end - y0) * (1.0 - np.exp(-2.0 * t[m1] / t_slow_end))
    m2 = (t > t_slow_end) & (t <= plateau_until)
    out[m2] = plateau_level + 0.015 * np.sin(t[m2] / 220.0)
    y_after_plateau = float(plateau_level + 0.015 * np.sin(plateau_until / 220.0))
    m3 = t > plateau_until
    span = max(t_peak - plateau_until, 1.0)
    out[m3] = y_after_plateau + (y_peak - y_after_plateau) * (
        1.0 - np.exp(-rise_strength * (t[m3] - plateau_until) / span)
    )
    return np.minimum(out, y_peak * 1.02)


def _piecewise_sac(
    t: np.ndarray,
    style: CurveStyle,
    t_max: float,
    decay_k: float,
    y_end: float | None = None,
) -> np.ndarray:
    t_peak = style.t_peak
    rise = _sac_rise_shape(
        np.minimum(t, t_peak),
        style.y0,
        style.y_peak,
        t_peak,
        style.sac_slow_rise_frac,
        style.sac_plateau_level,
        style.sac_plateau_until,
        style.rise_strength,
    )
    full = np.empty_like(t, dtype=np.float64)
    early = t <= t_peak
    full[early] = rise[early]
    y_peak = style.y_peak
    if y_end is None:
        y_end = style.y_end
    td = np.maximum(0.0, t[~early] - t_peak)
    full[~early] = y_end + (y_peak - y_end) * np.exp(
        -decay_k * td / max(t_max - t_peak, 1.0)
    )
    return full


def _piecewise_pretrain(
    t: np.ndarray,
    y_low: float,
    y_plateau: float,
    t_plateau: float,
    y_end: float,
    t_max: float,
    decay_k: float,
) -> np.ndarray:
    """更低起点 -> 略升（分布偏移）-> 指数衰减至 y_end。"""
    t = np.asarray(t, dtype=np.float64)
    out = np.empty_like(t)
    early = t <= t_plateau
    out[early] = y_low + (y_plateau - y_low) * (
        1.0 - np.exp(-3.5 * t[early] / max(t_plateau, 1.0))
    )
    td = np.maximum(0.0, t[~early] - t_plateau)
    out[~early] = y_end + (y_plateau - y_end) * np.exp(
        -decay_k * td / max(t_max - t_plateau, 1.0)
    )
    return out


def mean_curve(
    t: np.ndarray,
    style: CurveStyle,
    t_max: float,
    reward_shaping: bool,
    pretrain_bc: bool,
    gnn: str,
    algo: str,
    bc_peak_lag: float = BC_PEAK_LAG_STEPS,
) -> np.ndarray:
    sage_factor = 1.18 if gnn.lower() == "graphsage" else 1.0
    decay_k_base = style.decay_k * sage_factor
    high_start = 0.405

    def smooth_decay(start_v: float, end_v: float, k: float) -> np.ndarray:
        # 从高起点平滑下降到给定收敛值（无突变）
        tau = np.maximum(t, 0.0) / max(t_max, 1.0)
        return end_v + (start_v - end_v) * np.exp(-k * tau)

    base_y_end = 0.23
    base = smooth_decay(high_start, base_y_end, decay_k_base * 1.02)

    # base
    if not reward_shaping and not pretrain_bc:
        return base

    # +reward_shaping：与 base 同高起点、无突变，下降略快；收敛值保持不变
    rs_decay_k = decay_k_base * 1.12
    rs_y_end = 0.22
    if reward_shaping and not pretrain_bc:
        return smooth_decay(high_start, rs_y_end, rs_decay_k * 1.02)

    # +pretrain_bc：抖动直线（无明显下降趋势）
    bc_start = style.y0 - 0.055
    pretrain_flat = (
        np.full_like(t, bc_start, dtype=np.float64)
        + 0.0035 * np.sin(t / 160.0)
        + 0.0015 * np.sin(t / 37.0)
    )
    if pretrain_bc and not reward_shaping:
        return pretrain_flat

    # +pretrain_bc+reward_shaping：
    # 起点沿用 pretrain_bc，1000-2000 步突增（仍低于 base），随后略慢于 rs 衰减，最终最低
    jump_center = 1450.0
    jump_scale = 90.0
    jump_target = np.minimum(np.full_like(t, 0.28, dtype=np.float64), base - 0.005)
    jump_ratio = 1.0 / (1.0 + np.exp(-(t - jump_center) / jump_scale))
    combo = pretrain_flat + jump_ratio * np.maximum(jump_target - bc_start, 0.0)

    decay_start = 2000.0
    combo_y_end = 0.170
    combo_decay_k = rs_decay_k * 0.82
    m = t >= decay_start
    if np.any(m):
        v0 = float(combo[np.argmax(m)])
        td = t[m] - decay_start
        combo[m] = combo_y_end + (v0 - combo_y_end) * np.exp(
            -combo_decay_k * td / max(t_max - decay_start, 1.0)
        )
    return combo


def noisy_auc(mean: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    n = mean + rng.normal(0.0, noise_std, size=mean.shape)
    return np.clip(n, 0.12, 0.48)


def styles(decay_start: float = DECAY_START_STEP) -> dict[Tuple[str, str], CurveStyle]:
    t0 = float(decay_start)
    return {
        ("gcn", "dqn"): CurveStyle(
            y0=0.24,
            y_peak=0.42,
            t_peak=t0,
            y_end=0.26,
            decay_k=2.15,
            rise_strength=5.2,
            noise_std=0.034,
        ),
        ("gcn", "sac"): CurveStyle(
            y0=0.24,
            y_peak=0.31,
            t_peak=t0,
            y_end=0.245,
            decay_k=2.85,
            rise_strength=3.6,
            noise_std=0.014,
            sac_slow_rise_frac=0.35,
            sac_plateau_level=0.265,
            sac_plateau_until=1100.0,
        ),
        ("graphsage", "dqn"): CurveStyle(
            y0=0.24,
            y_peak=0.41,
            t_peak=t0,
            y_end=0.255,
            decay_k=2.15,
            rise_strength=5.4,
            noise_std=0.032,
        ),
        ("graphsage", "sac"): CurveStyle(
            y0=0.24,
            y_peak=0.30,
            t_peak=t0,
            y_end=0.24,
            decay_k=2.95,
            rise_strength=3.4,
            noise_std=0.013,
            sac_slow_rise_frac=0.32,
            sac_plateau_level=0.26,
            sac_plateau_until=1050.0,
        ),
    }


VARIANTS: list[tuple[str, bool, bool]] = [
    ("base", False, False),
    ("rs", True, False),
    ("pretrain_bc", False, True),
    ("pretrain_bc_rs", True, True),
]


def run_name(gnn: str, algo: str, variant: str) -> str:
    return f"synthetic/synthetic_{gnn}_{algo}_{variant}"


def main() -> None:
    p = argparse.ArgumentParser(description="Write synthetic train/AUC TensorBoard runs.")
    p.add_argument(
        "--root",
        type=str,
        default="runs",
        help="TensorBoard log root (same convention as teacher scripts: runs/<name>).",
    )
    p.add_argument("--t_max", type=int, default=20000, help="Last global_step (inclusive span).")
    p.add_argument("--step", type=int, default=50, help="Log every this many steps (match teacher).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gnn", type=str, default="gcn", choices=["gcn", "graphsage"])
    p.add_argument("--algo", type=str, default="dqn", choices=["dqn", "sac"])
    p.add_argument(
        "--decay_start",
        type=float,
        default=DECAY_START_STEP,
        help="Global step where curves peak and begin decay (default: match learning_starts=2000).",
    )
    p.add_argument(
        "--bc_peak_lag",
        type=float,
        default=BC_PEAK_LAG_STEPS,
        help="Extra steps before peak+decay for BC/pretrain runs (distribution shift bump).",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    # 与 teacher 一致：(global_step+1) % 50 == 0 -> 49,99,...
    steps = np.arange(args.step - 1, args.t_max + 1, args.step, dtype=np.int64)
    t = steps.astype(np.float64)

    style_map = styles(float(args.decay_start))
    os.makedirs(args.root, exist_ok=True)

    gnn = args.gnn.lower()
    algo = args.algo.lower()
    st = style_map[(gnn, algo)]
    for variant, rs, pt in VARIANTS:
        mean = mean_curve(
            t,
            st,
            float(args.t_max),
            rs,
            pt,
            gnn,
            algo,
            bc_peak_lag=float(args.bc_peak_lag),
        )
        raw = noisy_auc(mean, st.noise_std, rng)
        name = run_name(gnn, algo, variant)
        logdir = os.path.join(args.root, name)
        with SummaryWriter(logdir) as w:
            for gi, v in zip(steps, raw):
                w.add_scalar("train/AUC", float(v), int(gi))
        print(f"Wrote {len(steps)} points -> {logdir}")


if __name__ == "__main__":
    main()
