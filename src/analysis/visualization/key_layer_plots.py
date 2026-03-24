"""Plotting utilities for key-layer analysis outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from analysis.key_layers import CosineCurves

# --- 样式常量（美化用）---
STYLE_FONT = {'family': 'sans-serif', 'weight': 'normal', 'size': 8}
plt.rc('font', **STYLE_FONT)
# 配色：N-N / N-R / Angular，区分清晰且协调
COLOR_NN = '#C45C2A'       # 暖橙
COLOR_NR = '#2D7D6E'       # 青绿
COLOR_ANGLE = '#4A6FA5'    # 蓝灰
FILL_NN = '#E8B89A'        # 浅橙
FILL_NR = '#A8D5C8'        # 浅青绿
GRID_COLOR = '#E0E0E0'
SPINE_COLOR = '#444444'
# -----------------

def _normalise_path(path: Path | str) -> Path:
    resolved = Path(path)
    if not resolved.parent.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

def plot_dual_key_layer_analysis(
    baseline1: CosineCurves,
    defense1: CosineCurves,
    name1: str,
    baseline2: CosineCurves,
    defense2: CosineCurves,
    name2: str,
    output_path: Path | str,
    # 保持您设定的尺寸 (3.5, 3.0)
    figsize: Tuple[float, float] = (3.5, 3.0),
    dpi: int = 300,
    gap_ylabel: str = "Angular Gap (°)",
) -> Path:
    """
    绘制两个防御模型的 angular gap 视角比较图。

    - 每个模型占一列
    - 每列包含两行：
      上行：Baseline Gap / Defense Gap 双曲线 + 阴影误差带
      下行：gap_diff = gap_def - gap_base 单曲线
    """
    output = _normalise_path(output_path)
    
    # 2x2：行 = {Gap, GapDiff}，列 = {Model1, Model2}
    # 不同防御模型（不同列）在幅度上差异可能很大（例如 safety 更明显）。
    # 因此这里不共享 Y 轴，让每列自动缩放刻度，避免 privacy 曲线被“压扁”。
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=False)
    fig.patch.set_facecolor('white')
    
    configs = [
        {"baseline": baseline1, "defense": defense1, "name": name1, "col": 0},
        {"baseline": baseline2, "defense": defense2, "name": name2, "col": 1},
    ]

    line_width = 2.0
    fill_alpha = 0.35

    def _style_axis(ax: Axes, show_xlabel: bool = False) -> None:
        ax.set_facecolor('white')
        ax.tick_params(axis='both', which='major', labelsize=9, colors=SPINE_COLOR)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(SPINE_COLOR)
        ax.spines['bottom'].set_color(SPINE_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.6, linestyle='-', alpha=0.8)
        ax.set_axisbelow(True)
        if show_xlabel:
            ax.set_xlabel("Layer Index", fontsize=8, color=SPINE_COLOR)

    for config in configs:
        baseline: CosineCurves = config["baseline"]
        defense: CosineCurves = config["defense"]
        col = config["col"]
        title_name = config["name"]

        num_layers = len(defense.mu_nn)
        layers = list(range(num_layers))

        # --- 计算 Gap 相关量 ---
        gap_base = np.asarray(baseline.delta_phi)
        gap_def = np.asarray(defense.delta_phi)

        # 角度层面的误差带：简单用 N/R 角度 std 合成
        base_std = np.sqrt(
            np.square(np.asarray(baseline.angle_std_nn))
            + np.square(np.asarray(baseline.angle_std_nr))
        )
        def_std = np.sqrt(
            np.square(np.asarray(defense.angle_std_nn))
            + np.square(np.asarray(defense.angle_std_nr))
        )

        # --- Row 0: Baseline / Defense Angular Gap ---
        ax_gap = axes[0, col]
        ax_gap.plot(layers, gap_base, label="Baseline Gap", color=COLOR_NN, linewidth=line_width)
        ax_gap.plot(layers, gap_def, label="Defense Gap", color=COLOR_NR, linewidth=line_width)

        ax_gap.fill_between(
            layers,
            gap_base - base_std,
            gap_base + base_std,
            color=FILL_NN,
            alpha=fill_alpha,
        )
        ax_gap.fill_between(
            layers,
            gap_def - def_std,
            gap_def + def_std,
            color=FILL_NR,
            alpha=fill_alpha,
        )

        if col == 0:
            ax_gap.set_ylabel(gap_ylabel, fontsize=8, color=SPINE_COLOR)
        ax_gap.set_title(title_name, fontsize=8, fontweight='bold', pad=8, color=SPINE_COLOR)
        ax_gap.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.95, edgecolor=GRID_COLOR)
        ax_gap.yaxis.set_major_locator(MaxNLocator(nbins=5))
        _style_axis(ax_gap, show_xlabel=False)

        # --- Row 1: Gap Difference (Defense - Baseline) ---
        ax_diff = axes[1, col]
        gap_diff = gap_def - gap_base
        ax_diff.plot(layers, gap_diff, label="ΔGap (Def − Base)", color=COLOR_ANGLE, linewidth=line_width)
        if col == 0:
            ax_diff.set_ylabel("Gap Difference (°)", fontsize=8, color=SPINE_COLOR)
        ax_diff.yaxis.set_major_locator(MaxNLocator(nbins=5))
        _style_axis(ax_diff, show_xlabel=True)

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(wspace=0.08, hspace=0.22)

    fig.savefig(output, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Comparison plot saved to {output}")
    return output