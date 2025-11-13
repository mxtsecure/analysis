"""Visualization helpers for granular parameter delta analysis."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import matplotlib.pyplot as plt
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from analysis.parameter_deltas import ParameterDeltaSummary


def _ensure_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _layer_sort_key(label: str) -> tuple[int, str]:
    try:
        return (int(label.split(".")[-1]), label)
    except (IndexError, ValueError):
        return (1_000_000, label)


def plot_module_heatmap(
    modules_df: pd.DataFrame,
    output_path: Path,
    *,
    metric: str = "mean_abs",
    cmap: str = "magma",
    title: str | None = None,
) -> None:
    """Render a heatmap of module-level statistics across layers."""

    if modules_df.empty:
        raise ValueError("modules_df must not be empty")
    if metric not in modules_df.columns:
        raise KeyError(f"Metric '{metric}' not found in modules_df")

    modules_df = modules_df.copy()
    modules_df["module_name"] = modules_df["module"].str.split(".").str[-1]
    modules_df["layer_label"] = modules_df["module"].str.extract(
        r"(model\.layers\.[0-9]+)", expand=False
    )
    pivot = modules_df.pivot_table(
        values=metric,
        index="module_name",
        columns="layer_label",
        aggfunc="mean",
    )
    pivot = pivot.sort_index(axis=0)
    ordered_columns = sorted(pivot.columns, key=_layer_sort_key)
    pivot = pivot.loc[:, ordered_columns]

    _ensure_output_dir(output_path)
    fig, ax = plt.subplots(figsize=(0.6 * pivot.shape[1] + 4, 0.5 * pivot.shape[0] + 2))
    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Module")
    ax.set_title(title or f"Module heatmap ({metric})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_layer_curves(
    layers_df: pd.DataFrame,
    output_path: Path,
    *,
    metrics: Iterable[str] = ("mean_abs", "mean_l2", "mean_abs_rel"),
    title: str | None = None,
) -> None:
    """Render line plots for the requested metrics across layers."""

    if layers_df.empty:
        raise ValueError("layers_df must not be empty")

    layers_df = layers_df.copy()
    if "layer_index" in layers_df.columns:
        layers_df = layers_df.sort_values("layer_index")
    x = layers_df["layer"].tolist()

    _ensure_output_dir(output_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    for metric in metrics:
        if metric not in layers_df.columns:
            raise KeyError(f"Metric '{metric}' not found in layers_df")
        ax.plot(x, layers_df[metric], marker="o", label=metric)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    ax.set_title(title or "Layer-wise parameter deltas")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def export_summary_artifacts(
    summary: "ParameterDeltaSummary",
    output_dir: Path,
    *,
    heatmap_metrics: Iterable[str] = ("mean_abs", "mean_abs_rel"),
    curve_metrics: Iterable[str] = ("mean_abs", "mean_l2", "mean_abs_rel"),
) -> dict[str, Path]:
    """Save CSV tables and plots for a :class:`ParameterDeltaSummary`."""

    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    summary.parameters.to_csv(output_dir / "parameters.csv", index=False)
    summary.layers.to_csv(output_dir / "layers.csv", index=False)
    summary.modules.to_csv(output_dir / "modules.csv", index=False)
    summary.submodules.to_csv(output_dir / "submodules.csv", index=False)

    artifacts["parameters_csv"] = output_dir / "parameters.csv"
    artifacts["layers_csv"] = output_dir / "layers.csv"
    artifacts["modules_csv"] = output_dir / "modules.csv"
    artifacts["submodules_csv"] = output_dir / "submodules.csv"

    if not summary.modules.empty:
        for metric in heatmap_metrics:
            heatmap_path = output_dir / f"module_heatmap_{metric}.png"
            plot_module_heatmap(summary.modules, heatmap_path, metric=metric)
            artifacts[f"module_heatmap_{metric}"] = heatmap_path

    if not summary.layers.empty:
        curves_path = output_dir / "layer_curves.png"
        plot_layer_curves(summary.layers, curves_path, metrics=curve_metrics)
        artifacts["layer_curves"] = curves_path

    return artifacts
