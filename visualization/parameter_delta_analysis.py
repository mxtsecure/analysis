#!/usr/bin/env python
"""Analyze parameter deltas between original and defended model checkpoints."""

import argparse
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoModelForCausalLM


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load model weights using Hugging Face's AutoModelForCausalLM interface."""

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    try:
        return {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    finally:
        del model


def infer_layer_component(parameter_name: str) -> Tuple[int, str]:
    """Infer (layer_index, component_type) from a parameter name.

    Parameters falling outside the recognised naming pattern are grouped
    under component "other" and assigned to layer -1.
    """

    tokens = parameter_name.split(".")
    try:
        layer_idx = tokens.index("layers")
        layer_number = int(tokens[layer_idx + 1])
    except ValueError:
        return -1, "other"
    except (IndexError, ValueError):
        return -1, "other"

    if "self_attn" in tokens:
        return layer_number, "attention"
    if "mlp" in tokens:
        return layer_number, "mlp"
    return layer_number, "other"


def compute_parameter_deltas(
    original: Dict[str, torch.Tensor], defended: Dict[str, torch.Tensor]
) -> pd.DataFrame:
    """Compute parameter delta statistics between two state dicts."""

    rows = []
    shared_params = sorted(set(original.keys()).intersection(defended.keys()))
    for name in shared_params:
        orig_tensor = original[name]
        def_tensor = defended[name]
        if orig_tensor.shape != def_tensor.shape:
            continue
        if def_tensor.dtype != orig_tensor.dtype:
            # Align dtypes for consistent comparisons.
            def_tensor = def_tensor.to(orig_tensor.dtype)
        work_tensor = def_tensor.to(torch.float32) - orig_tensor.to(torch.float32)
        delta = work_tensor
        layer, component = infer_layer_component(name)
        rows.append(
            {
                "parameter": name,
                "layer": layer,
                "component": component,
                "numel": delta.numel(),
                "l2": torch.norm(delta).item(),
                "mean_abs": delta.abs().mean().item(),
                "max_abs": delta.abs().max().item(),
            }
        )

    return pd.DataFrame(rows)


def aggregate_by_layer_component(
    df: pd.DataFrame, metric: str, agg: str
) -> pd.DataFrame:
    """Aggregate parameter statistics by layer and component type."""

    if df.empty:
        return pd.DataFrame(columns=["layer", "component", metric])

    grouped = (
        df.groupby(["layer", "component"], as_index=False)[metric]
        .aggregate(agg)
        .rename(columns={metric: f"{metric}_{agg}"})
    )
    return grouped.sort_values(by=["layer", "component"]).reset_index(drop=True)


def plot_heatmap(pivot: pd.DataFrame, metric_label: str, output_path: str) -> None:
    """Plot a heatmap of layer/component metric values."""

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot.index) * 0.4)))
    heatmap = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Component")
    ax.set_ylabel("Layer")
    ax.set_title(f"Parameter Delta {metric_label} (Heatmap)")
    fig.colorbar(heatmap, ax=ax, label=metric_label)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_line_chart(df: pd.DataFrame, metric_label: str, output_path: str) -> None:
    """Plot line chart of metric values per layer for each component."""

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for component, group in df.groupby("component"):
        sorted_group = group.sort_values("layer")
        ax.plot(
            sorted_group["layer"],
            sorted_group[metric_label],
            marker="o",
            label=component,
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Parameter Delta {metric_label} (Line Plot)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main(args: argparse.Namespace) -> None:
    original_state = load_state_dict(args.original_model)
    defended_state = load_state_dict(args.defended_model)

    df = compute_parameter_deltas(original_state, defended_state)
    if df.empty:
        print("No shared parameters with matching shapes were found.")
        return

    metric_label = args.metric
    agg_label = args.agg
    aggregated = aggregate_by_layer_component(df, metric_label, agg_label)
    metric_column = f"{metric_label}_{agg_label}"

    ensure_output_dir(args.output_dir)
    csv_path = os.path.join(args.output_dir, "parameter_delta_summary.csv")
    aggregated.to_csv(csv_path, index=False)
    print(f"Saved aggregated statistics to {csv_path}")

    pivot = aggregated.pivot_table(
        index="layer", columns="component", values=metric_column, fill_value=0.0
    )
    heatmap_path = os.path.join(args.output_dir, f"parameter_delta_heatmap_{metric_label}_{agg_label}.png")
    plot_heatmap(pivot, metric_column, heatmap_path)
    print(f"Saved heatmap to {heatmap_path}")

    line_chart_path = os.path.join(
        args.output_dir, f"parameter_delta_line_{metric_label}_{agg_label}.png"
    )
    plot_line_chart(aggregated, metric_column, line_chart_path)
    print(f"Saved line chart to {line_chart_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze parameter deltas between original and defended models."
    )
    parser.add_argument("original_model", help="Path to the original model checkpoint")
    parser.add_argument("defended_model", help="Path to the defended model checkpoint")
    parser.add_argument(
        "--output-dir",
        default="results/visualization/parameter_deltas",
        help="Directory to store the generated artifacts",
    )
    parser.add_argument(
        "--metric",
        default="l2",
        choices=["l2", "mean_abs", "max_abs"],
        help="Metric used to summarize parameter deltas",
    )
    parser.add_argument(
        "--agg",
        default="sum",
        choices=["sum", "mean", "max"],
        help="Aggregation function applied across parameters in each group",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())
