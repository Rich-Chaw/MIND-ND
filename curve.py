#!/usr/bin/env python3
"""
合成 train/AUC 曲线并写入 TensorBoard，便于与 dqn_teacher.py / sac_teacher.py 的
SummaryWriter 用法对齐（tag: train/AUC，步长与训练循环一致地取每 50 step 一点）。

四组：GCN+DQN、GCN+SAC、GraphSAGE+DQN、GraphSAGE+SAC；每组 4 条：
  base | +reward_shaping | +pretrain_bc | +pretrain_bc+reward_shaping

峰值与下降起点与 dqn_teacher / sac_teacher 默认 learning_starts 对齐（2000 步起更新网络，
无 BC 的合成曲线约在此时达峰并进入下降）。

含 BC/pretrain 的曲线峰值整体再后移约 1000–2000 步，表示微调初期因与专家分布不一致
而出现的短暂冲高后再下降。
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
    rs_factor = 1.48 if reward_shaping else 1.0
    sage_factor = 1.32 if gnn.lower() == "graphsage" else 1.0
    decay_k = style.decay_k * rs_factor * sage_factor

    if algo.lower() == "sac":
        base = _piecewise_sac(t, style, t_max, decay_k)
    else:
        base = _piecewise_rise_decay(
            t,
            style.y0,
            style.y_peak,
            style.t_peak,
            style.y_end,
            t_max,
            decay_k,
            style.rise_strength,
        )

    if not pretrain_bc:
        return base

    # BC / pretrain：末尾渐近 AUC 略低于无 BC（混合目标也用同一 y_end_bc，避免被拉回 base 的高位平台）
    bc_tail_drop = 0.017 if algo.lower() == "sac" else 0.026
    y_end_bc = max(0.17, style.y_end - bc_tail_drop)

    # 预训练/BC：起点更低，经分布偏移段后再达峰，峰值比无 BC 晚 bc_peak_lag 步，再衰减
    t_peak_bc = style.t_peak + float(bc_peak_lag)
    y_low = style.y0 - 0.055
    y_plateau = min(style.y_peak - 0.11, style.y0 + 0.065)
    tail = _piecewise_pretrain(
        t, y_low, y_plateau, t_peak_bc, y_end_bc, t_max, decay_k * 0.95
    )
    if algo.lower() == "sac":
        # t_peak 后移时同步后移 SAC 平台段终点，保持 slow / plateau / 末段爬升顺序
        style_bc = replace(
            style,
            t_peak=t_peak_bc,
            sac_plateau_until=style.sac_plateau_until + float(bc_peak_lag),
        )
        base_bc = _piecewise_sac(t, style_bc, t_max, decay_k, y_end=y_end_bc)
    else:
        base_bc = _piecewise_rise_decay(
            t,
            style.y0,
            style.y_peak,
            t_peak_bc,
            y_end_bc,
            t_max,
            decay_k,
            style.rise_strength,
        )
    alpha = np.clip((t - t_peak_bc) / 5000.0, 0.0, 1.0)
    return (1.0 - alpha) * tail + alpha * base_bc


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
    return f"synthetic_{gnn}_{algo}_{variant}"


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

    for gnn in ("gcn", "graphsage"):
        for algo in ("dqn", "sac"):
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
