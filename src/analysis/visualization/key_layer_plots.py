"""Plotting utilities for key-layer analysis outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from analysis.key_layers import KeyLayerAnalysisResult

LayerRange = Tuple[int, int]


def _normalise_path(path: Path | str) -> Path:
    resolved = Path(path)
    if not resolved.parent.exists():
        resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _apply_interval_shading(
    axes: Sequence[Axes],
    representational: LayerRange,
    key_interval: Optional[LayerRange],
    num_layers: int,
) -> None:
    rep_start, rep_end = representational
    rep_end = min(rep_end, num_layers - 1)
    for ax in axes:
        ax.axvspan(rep_start - 0.5, rep_end + 0.5, color="tab:blue", alpha=0.08, label=None)
        if key_interval is not None:
            key_start, key_end = key_interval
            key_end = min(key_end, num_layers - 1)
            ax.axvspan(key_start - 0.5, key_end + 0.5, color="tab:orange", alpha=0.15, label=None)


def _apply_vertical_markers(
    axes: Sequence[Axes],
    layers: Iterable[int],
    linestyle: str = "--",
    color: str = "tab:red",
    alpha: float = 0.6,
) -> None:
    for layer in layers:
        for ax in axes:
            ax.axvline(layer, linestyle=linestyle, color=color, alpha=alpha)


def plot_key_layer_analysis(
    result: KeyLayerAnalysisResult,
    output_path: Path | str,
    *,
    title: Optional[str] = None,
    critical_layers: Optional[Iterable[int]] = None,
    figsize: Tuple[float, float] = (12.0, 8.0),
    dpi: int = 300,
) -> Path:
    """Render a stacked summary figure for the key-layer analysis."""

    cosine = result.cosine
    params = result.parameters
    num_layers = len(cosine.mu_nn)
    layers = list(range(num_layers))

    output = _normalise_path(output_path)

    fig: Figure
    axes: Sequence[Axes]
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)

    # Panel 1: cosine similarity statistics with ±1σ bands.
    ax_cos = axes[0]
    ax_cos.plot(layers, cosine.mu_nn, label="N–N cosine", color="tab:blue")
    ax_cos.plot(layers, cosine.mu_nr, label="N–M cosine", color="tab:green")
    lower_nn = [m - s for m, s in zip(cosine.mu_nn, cosine.std_nn)]
    upper_nn = [m + s for m, s in zip(cosine.mu_nn, cosine.std_nn)]
    lower_nr = [m - s for m, s in zip(cosine.mu_nr, cosine.std_nr)]
    upper_nr = [m + s for m, s in zip(cosine.mu_nr, cosine.std_nr)]
    ax_cos.fill_between(layers, lower_nn, upper_nn, color="tab:blue", alpha=0.15)
    ax_cos.fill_between(layers, lower_nr, upper_nr, color="tab:green", alpha=0.15)
    ax_cos.set_ylabel("Cosine similarity")
    ax_cos.set_title("Representation alignment")
    ax_cos.legend(loc="lower left")

    # Panel 2: angular differences (raw and smoothed).
    ax_angle = axes[1]
    ax_angle.plot(layers, cosine.delta_phi, label="Δφ (mean)", color="tab:red", alpha=0.6)
    ax_angle.plot(
        layers,
        cosine.smooth_delta_phi,
        label="Δφ (smoothed)",
        color="tab:red",
        linewidth=2.0,
    )
    ax_angle.set_ylabel("Angle diff (deg)")
    ax_angle.set_title("Angular divergence")
    ax_angle.legend(loc="upper left")

    # Panel 3: parameter differences (attention vs MLP vs total).
    ax_params = axes[2]
    ax_params.plot(layers, params.s_attn, label="|Δθ| attention", color="tab:purple")
    ax_params.plot(layers, params.s_mlp, label="|Δθ| MLP", color="tab:orange")
    ax_params.plot(layers, params.s_total, label="|Δθ| total", color="tab:gray", linewidth=2.0)
    ax_params.set_ylabel("Mean |Δθ|")
    ax_params.set_xlabel("Layer index")
    ax_params.set_title("Parameter drift")
    ax_params.legend(loc="upper left")

    _apply_interval_shading(axes, result.intervals.representational, result.intervals.key, num_layers)

    if critical_layers:
        _apply_vertical_markers(axes, critical_layers)

    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))

    if title:
        fig.suptitle(title)

    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output

