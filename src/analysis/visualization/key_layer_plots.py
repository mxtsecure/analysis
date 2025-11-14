"""Plotting utilities for key-layer analysis outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

# 假设这些类和类型是可用的
from analysis.key_layers import KeyLayerAnalysisResult, CosineCurves, ParameterCurves, KeyLayerIntervals 

LayerRange = Tuple[int, int]
# --- 样式常量 ---
STYLE_FONT = {'family': 'sans-serif', 'weight': 'normal'}
plt.rc('font', **STYLE_FONT)
# -----------------


def _normalise_path(path: Path | str) -> Path:
    resolved = Path(path)
    if not resolved.parent.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _apply_interval_shading(
    axes: Sequence[Axes],
    representational_segments: Sequence[LayerRange],
    parameter_segments: Sequence[LayerRange],
    num_layers: int,
) -> None:
    """Apply vertical shading for representational and parameter spans."""
    for ax in axes:
        for rep_start, rep_end in representational_segments:
            rep_end_safe = min(rep_end, num_layers - 1)
            ax.axvspan(
                rep_start - 0.5,
                rep_end_safe + 0.5,
                color="tab:blue",
                alpha=0.08,
                label=None,
            )
        for par_start, par_end in parameter_segments:
            par_end_safe = min(par_end, num_layers - 1)
            ax.axvspan(
                par_start - 0.5,
                par_end_safe + 0.5,
                color="tab:orange",
                alpha=0.12,
                label=None,
            )


def _apply_vertical_markers(
    axes: Sequence[Axes],
    layers: Iterable[int],
    linestyle: str = "--",
    color: str = "red",  # 使用纯红
    alpha: float = 0.8, # 增加 alpha
) -> None:
    """Apply red dashed vertical lines at critical layers."""
    for layer in layers:
        for ax in axes:
            ax.axvline(layer, linestyle=linestyle, color=color, linewidth=1.5, alpha=alpha)


def plot_key_layer_analysis(
    result: KeyLayerAnalysisResult,
    output_path: Path | str,
    *,
    title: Optional[str] = None,
    critical_layers: Optional[Iterable[int]] = None,
    figsize: Tuple[float, float] = (8.0, 8.0), 
    dpi: int = 500,
) -> Path:
    """Render a stacked summary figure for the key-layer analysis with 3 panels and enhanced style."""

    cosine: CosineCurves = result.cosine
    params: ParameterCurves = result.parameters
    num_layers = len(cosine.mu_nn)
    layers = list(range(num_layers))

    output = _normalise_path(output_path)

    fig: Figure
    axes: Sequence[Axes]
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True) 

    # --- Panel 1: Cosine similarity statistics ---
    ax_cos = axes[0]
    
    # 曲线
    ax_cos.plot(layers, cosine.mu_nn, label="N–N Pairs", color="tab:orange", linewidth=2)
    ax_cos.plot(layers, cosine.mu_nr, label="N–M Pairs", color="tab:green", linewidth=2)
    
    lower_nn = [m - s for m, s in zip(cosine.mu_nn, cosine.std_nn)]
    upper_nn = [m + s for m, s in zip(cosine.mu_nn, cosine.std_nn)]
    lower_nr = [m - s for m, s in zip(cosine.mu_nr, cosine.std_nr)]
    upper_nr = [m + s for m, s in zip(cosine.mu_nr, cosine.std_nr)]
    
    ax_cos.fill_between(layers, lower_nn, upper_nn, color='bisque', alpha=0.45)
    ax_cos.fill_between(layers, lower_nr, upper_nr, color='lightgreen', alpha=0.25)
    
    # 装饰
    ax_cos.set_ylabel("Cos_sim Value", fontsize='large')
    ax_cos.tick_params(axis='both', labelsize='large')
    ax_cos.legend(loc="upper right", fontsize='medium')
    ax_cos.grid(True, linewidth=0.8, linestyle='--')
    ax_cos.set_title("Representation alignment", fontsize='large')

    # --- Panel 2: Angular differences (Δφ) ---
    ax_angle = axes[1]
    
    # 只绘制原始 Δφ 曲线 (Δφ = N-M angle - N-N angle)
    ax_angle.plot(layers, cosine.delta_phi, label="Mean Angular Difference", 
                  color="tab:blue", linewidth=2.0)
    
    # 装饰
    ax_angle.set_ylabel("Angle Degree Value", fontsize='large')
    ax_angle.tick_params(axis='both', labelsize='large')
    ax_angle.legend(loc="upper right", fontsize='medium')
    ax_angle.grid(True, linewidth=0.8, linestyle='--')
    ax_angle.set_title("Angular divergence", fontsize='large')

    # --- Panel 3: Parameter differences (MAD) ---
    ax_params = axes[2]
    
    # 绘制参数差异曲线 (MAD)
    ax_params.plot(layers, params.s_attn, label="|Δθ| attention", color="tab:purple")
    ax_params.plot(layers, params.s_mlp, label="|Δθ| MLP", color="tab:orange")
    ax_params.plot(layers, params.s_total, label="|Δθ| total", color="tab:gray", linewidth=2.0)
    
    # 装饰
    ax_params.set_ylabel("Mean |Δθ|", fontsize='large')
    ax_params.set_xlabel("Layer index", fontsize='large')
    ax_params.tick_params(axis='both', labelsize='large')
    ax_params.legend(loc="upper right", fontsize='medium')
    ax_params.grid(True, linewidth=0.8, linestyle='--')
    ax_params.set_title("Parameter drift", fontsize='large')

    # --- 应用区间和标记 ---
    
    rep_segments = list(result.intervals.representational)
    parameter_segments = list(result.intervals.parameter_spans)
    
    # _apply_interval_shading(axes, rep_segments, parameter_segments, num_layers)

    # if critical_layers:
    #     _apply_vertical_markers(axes, critical_layers)

    # X 轴整数刻度设置
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # --- 布局与保存 ---
    
    # 调整布局以适应标题
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 整体标题
    if title:
        fig.suptitle(title, fontsize='x-large', y=0.98) 
    overlap_segments = []
    for rep_start, rep_end in rep_segments:
        for span_start, span_end in parameter_segments:
            overlap_start = max(rep_start, span_start)
            overlap_end = min(rep_end, span_end)
            if overlap_start <= overlap_end:
                overlap_segments.append((overlap_start, overlap_end))


    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return output