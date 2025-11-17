"""Track how weight deltas evolve inside a conflict subspace across checkpoints."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.concept_vectors import aggregate_difference
from data.datasets import build_dual_dataset
from models.activations import compute_activation_difference, forward_dataset


@dataclass
class LayerDirections:
    """Container storing the conflict subspace directions for a layer."""

    name: str
    safe_vector: torch.Tensor
    priv_vector: torch.Tensor


@dataclass
class CheckpointSpec:
    """Represents a checkpoint along a defence trajectory."""

    path: Path
    step: float
    label: str
    defense: str
    baseline: str


def _infer_layer_names(model_path: str | Path) -> List[str]:
    config = AutoConfig.from_pretrained(model_path)
    total_layers = getattr(config, "num_hidden_layers", None)
    if total_layers is None:
        raise ValueError(
            "Unable to determine transformer layer count from the provided base model"
        )
    return [f"model.layers.{idx}" for idx in range(int(total_layers))]


def _ensure_activation_mapping(value, layers: Sequence[str]):
    if isinstance(value, dict):
        return value
    if len(layers) != 1:
        raise ValueError(
            "Expected multiple layers to yield an activation mapping but received a single tensor"
        )
    return {layers[0]: value}


def _forward_with_hooks(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    layers: Sequence[str],
    device: torch.device,
    description: str,
    *,
    max_batches: int = 4,
):
    activations = forward_dataset(
        model,
        dataloader,
        layers,
        device,
        description,
        max_batches=max_batches,
    )
    return _ensure_activation_mapping(activations, layers)


def _compute_concept_vectors(
    base_path: str | Path,
    defense_path: str | Path,
    dataloader: DataLoader,
    layers: Sequence[str],
    device: torch.device,
    description: str,
    *,
    max_batches: int = 4,
) -> Dict[str, torch.Tensor]:
    base_model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float16).to(
        device
    )
    defense_model = AutoModelForCausalLM.from_pretrained(
        defense_path, torch_dtype=torch.float16
    ).to(device)
    try:
        base_acts = _forward_with_hooks(
            base_model,
            dataloader,
            layers,
            device,
            f"{description} (base)",
            max_batches=max_batches,
        )
        defense_acts = _forward_with_hooks(
            defense_model,
            dataloader,
            layers,
            device,
            f"{description} (defense)",
            max_batches=max_batches,
        )
    finally:
        base_model.cpu()
        defense_model.cpu()
        del base_model
        del defense_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    vectors: Dict[str, torch.Tensor] = {}
    for layer in layers:
        delta = compute_activation_difference(base_acts[layer], defense_acts[layer])
        concept = aggregate_difference(delta, method="mean")
        direction = concept.direction.detach().cpu().to(torch.float64)
        norm = torch.norm(direction)
        if norm.item() == 0:
            raise ValueError(f"Concept vector for layer {layer} is zero")
        vectors[layer] = direction / norm
    return vectors


def _derive_defense_label(paths: Sequence[str], fallback: str) -> str:
    for raw in reversed(paths):
        name = Path(raw).name
        if name:
            return name
    return fallback


def _build_simple_checkpoint_list(
    paths: Sequence[str],
    defense: str,
    start_step: float,
    step_increment: float,
    baseline: str,
) -> tuple[List[CheckpointSpec], float]:
    entries = []
    current_step = start_step
    for path in paths:
        current_step += step_increment
        p = Path(path)
        entries.append(
            CheckpointSpec(
                path=p, step=current_step, label=p.name, defense=defense, baseline=baseline
            )
        )
    return entries, current_step


def _select_layer_parameters(state_dict_keys: Iterable[str], layer: str) -> List[str]:
    prefix = f"{layer}."
    selected = [name for name in state_dict_keys if name == layer or name.startswith(prefix)]
    selected.sort()
    if not selected:
        raise ValueError(f"Layer '{layer}' was not found in the checkpoint state dict")
    return selected


def _build_risk_dataloaders(
    risk_dir: Path,
    tokenizer,
    batch_size: int = 4,
) -> tuple[DataLoader, DataLoader]:
    malicious_path = risk_dir / "safety.jsonl"
    privacy_path = risk_dir / "privacy.jsonl"
    normal_path = risk_dir / "normal.jsonl"
    for path in [normal_path, malicious_path, privacy_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {path}")
    datasets = build_dual_dataset(
        normal_path=normal_path,
        malicious_path=malicious_path,
        tokenizer=tokenizer,
        privacy_path=privacy_path,
    )
    if datasets.privacy is None:
        raise ValueError("Privacy dataset is required to compute concept vectors")
    malicious_loader = DataLoader(datasets.malicious, batch_size=batch_size)
    privacy_loader = DataLoader(datasets.privacy, batch_size=batch_size)
    return malicious_loader, privacy_loader


def _flatten_parameters(state: Mapping[str, torch.Tensor], names: Sequence[str]) -> torch.Tensor:
    vectors = []
    for name in names:
        tensor = state[name]
        if tensor.device.type != "cpu":
            tensor = tensor.detach().cpu()
        else:
            tensor = tensor.detach()
        vectors.append(tensor.reshape(-1).to(torch.float64))
    return torch.cat(vectors)


def _load_checkpoint_state(
    model_path: Path,
    *,
    include: Sequence[str] | None = None,
    torch_dtype: torch.dtype | None = None,
) -> Dict[str, torch.Tensor]:
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    if include is None:
        keys = state.keys()
    else:
        keys = include
    for name in keys:
        if name not in state:
            raise KeyError(f"Parameter '{name}' missing from checkpoint {model_path}")
        filtered[name] = state[name].detach().cpu().to(torch.float32).clone()
    del state
    model.cpu()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return filtered


def _compute_delta(
    base_vector: torch.Tensor,
    current_state: Mapping[str, torch.Tensor],
    names: Sequence[str],
) -> torch.Tensor:
    current_vector = torch.cat(
        [current_state[name].reshape(-1).to(torch.float64) for name in names]
    )
    if current_vector.shape != base_vector.shape:
        raise ValueError("Mismatched vector sizes when computing parameter delta")
    return current_vector - base_vector


def _compute_metrics(
    delta: torch.Tensor,
    safe_dir: torch.Tensor,
    priv_dir: torch.Tensor,
) -> tuple[float, float, float, float]:
    alpha_safe = float(torch.dot(delta, safe_dir).item())
    alpha_priv = float(torch.dot(delta, priv_dir).item())
    delta_norm = float(torch.norm(delta).item())
    if delta_norm == 0:
        angle = 0.0
    else:
        cosine = max(min(alpha_safe / delta_norm, 1.0), -1.0)
        angle = math.degrees(math.acos(cosine))
    return alpha_safe, alpha_priv, delta_norm, angle


def _plot_alpha_curves(
    df: pd.DataFrame,
    layer: str,
    output: Path,
    defense_order: Sequence[str],
) -> None:
    colors = {"Base": "#6c757d"}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, defense in enumerate(defense_order):
        colors[defense] = palette[idx % len(palette)]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.axhline(0, color="#dddddd", linewidth=1.0)
    for defense in ["Base", *defense_order]:
        subset = df[df["defense"] == defense].sort_values("step")
        if subset.empty:
            continue
        color = colors.get(defense, None)
        ax.plot(
            subset["step"],
            subset["alpha_safe"],
            label=f"{defense} α_safe (vs Base)",
            color=color,
            linestyle="-",
            marker="o",
        )
        ax.plot(
            subset["step"],
            subset["alpha_priv"],
            label=f"{defense} α_priv (vs Base)",
            color=color,
            linestyle="--",
            marker="o",
            alpha=0.8,
        )
        for baseline in subset["baseline"].unique():
            if baseline == "Base":
                continue
            rel = subset[subset["baseline"] == baseline]
            if rel.empty:
                continue
            ax.plot(
                rel["step"],
                rel["alpha_safe_relative"],
                label=f"{defense} α_safe (vs {baseline})",
                color=color,
                linestyle=":",
                marker="s",
            )
            ax.plot(
                rel["step"],
                rel["alpha_priv_relative"],
                label=f"{defense} α_priv (vs {baseline})",
                color=color,
                linestyle=(0, (3, 1, 1, 1)),
                marker="s",
                alpha=0.9,
            )
    ax.set_title(f"Layer {layer}: projection coefficients vs step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Projection coefficient")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_alpha_plane(
    df: pd.DataFrame,
    layer: str,
    output: Path,
    defense_order: Sequence[str],
) -> None:
    colors = {"Base": "#6c757d"}
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, defense in enumerate(defense_order):
        colors[defense] = palette[idx % len(palette)]
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.axhline(0, color="#dddddd", linewidth=1.0)
    ax.axvline(0, color="#dddddd", linewidth=1.0)
    for defense in ["Base", *defense_order]:
        subset = df[df["defense"] == defense].sort_values("step")
        if subset.empty:
            continue
        color = colors.get(defense)
        ax.plot(
            subset["alpha_safe"],
            subset["alpha_priv"],
            color=color,
            marker="o",
            label=f"{defense} trajectory (vs Base)",
        )
        for _, row in subset.iterrows():
            ax.annotate(
                f"{row['step']:.0f}",
                (row["alpha_safe"], row["alpha_priv"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
            )
        for baseline in subset["baseline"].unique():
            if baseline == "Base":
                continue
            rel = subset[subset["baseline"] == baseline]
            if rel.empty:
                continue
            ax.plot(
                rel["alpha_safe_relative"],
                rel["alpha_priv_relative"],
                color=color,
                marker="s",
                linestyle=":",
                label=f"{defense} trajectory (vs {baseline})",
            )
            for _, row in rel.iterrows():
                ax.annotate(
                    f"{row['step']:.0f}",
                    (row["alpha_safe_relative"], row["alpha_priv_relative"]),
                    textcoords="offset points",
                    xytext=(4, -6),
                    fontsize=7,
                    color=color,
                )
    ax.set_xlabel("α_safe")
    ax.set_ylabel("α_priv")
    ax.set_title(f"Layer {layer}: weight trajectory in conflict plane")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base checkpoint path")
    parser.add_argument(
        "--defense1",
        nargs="+",
        required=True,
        help="List of checkpoints belonging to the first defense trajectory",
    )
    parser.add_argument(
        "--defense2",
        nargs="+",
        required=True,
        help="List of checkpoints belonging to the second defense trajectory",
    )
    parser.add_argument(
        "--risk-data",
        type=Path,
        required=True,
        help="Directory containing risk datasets (normal.jsonl, safety.jsonl, privacy.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dynamic_conflict"),
        help="Directory where CSV and plots will be stored",
    )
    return parser.parse_args()


def _prepare_base_vectors(
    base_path: Path,
    layer_specs: Sequence[LayerDirections],
    torch_dtype: torch.dtype | None,
) -> tuple[Dict[str, torch.Tensor], Dict[str, List[str]]]:
    full_state = _load_checkpoint_state(base_path, torch_dtype=torch_dtype)
    layer_param_names: Dict[str, List[str]] = {}
    for spec in layer_specs:
        names = _select_layer_parameters(full_state.keys(), spec.name)
        layer_param_names[spec.name] = names
    needed = sorted({name for names in layer_param_names.values() for name in names})
    base_state = {name: full_state[name] for name in needed}
    del full_state
    base_vectors = {}
    for spec in layer_specs:
        vector = _flatten_parameters(base_state, layer_param_names[spec.name])
        if vector.numel() != spec.safe_vector.numel():
            raise ValueError(
                f"Safe direction for {spec.name} has {spec.safe_vector.numel()} elements "
                f"but the layer parameters flatten to {vector.numel()}"
            )
        if vector.numel() != spec.priv_vector.numel():
            raise ValueError(
                f"Privacy direction for {spec.name} has {spec.priv_vector.numel()} elements "
                f"but the layer parameters flatten to {vector.numel()}"
            )
        base_vectors[spec.name] = vector
    return base_vectors, layer_param_names


def _prepare_baseline_vectors(
    reference_paths: Mapping[str, Path],
    layer_param_names: Mapping[str, Sequence[str]],
    *,
    torch_dtype: torch.dtype | None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    include_keys = sorted({name for names in layer_param_names.values() for name in names})
    baseline_vectors: Dict[str, Dict[str, torch.Tensor]] = {}
    for label, path in reference_paths.items():
        state = _load_checkpoint_state(path, include=include_keys, torch_dtype=torch_dtype)
        layer_vectors: Dict[str, torch.Tensor] = {}
        for layer_name, names in layer_param_names.items():
            layer_vectors[layer_name] = _flatten_parameters(state, names)
        baseline_vectors[label] = layer_vectors
        del state
    return baseline_vectors


def _aggregate_records(
    checkpoints: Sequence[CheckpointSpec],
    base_vectors: Mapping[str, torch.Tensor],
    layer_param_names: Mapping[str, Sequence[str]],
    layer_specs: Mapping[str, LayerDirections],
    baseline_vectors: Mapping[str, Mapping[str, torch.Tensor]],
    torch_dtype: torch.dtype | None,
) -> List[dict]:
    include_keys = sorted({name for names in layer_param_names.values() for name in names})
    records: List[dict] = []
    for spec in checkpoints:
        state = _load_checkpoint_state(spec.path, include=include_keys, torch_dtype=torch_dtype)
        if spec.baseline not in baseline_vectors:
            raise KeyError(f"Baseline '{spec.baseline}' missing from provided vectors")
        for layer_name, names in layer_param_names.items():
            delta = _compute_delta(base_vectors[layer_name], state, names)
            layer_dir = layer_specs[layer_name]
            alpha_safe, alpha_priv, delta_norm, angle = _compute_metrics(
                delta, layer_dir.safe_vector, layer_dir.priv_vector
            )
            baseline_reference = baseline_vectors[spec.baseline]
            relative_delta = _compute_delta(baseline_reference[layer_name], state, names)
            (
                alpha_safe_rel,
                alpha_priv_rel,
                delta_norm_rel,
                angle_rel,
            ) = _compute_metrics(relative_delta, layer_dir.safe_vector, layer_dir.priv_vector)
            records.append(
                {
                    "layer": layer_name,
                    "defense": spec.defense,
                    "step": spec.step,
                    "checkpoint": str(spec.path),
                    "label": spec.label,
                    "baseline": spec.baseline,
                    "alpha_safe": alpha_safe,
                    "alpha_priv": alpha_priv,
                    "delta_norm": delta_norm,
                    "angle_deg": angle,
                    "alpha_safe_relative": alpha_safe_rel,
                    "alpha_priv_relative": alpha_priv_rel,
                    "delta_norm_relative": delta_norm_rel,
                    "angle_deg_relative": angle_rel,
                }
            )
        del state
    return records


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_names = _infer_layer_names(args.base)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    malicious_loader, privacy_loader = _build_risk_dataloaders(args.risk_data, tokenizer)
    safety_anchor = args.defense1[-1]
    privacy_anchor = args.defense2[-1]
    safety_vectors = _compute_concept_vectors(
        args.base,
        safety_anchor,
        malicious_loader,
        layer_names,
        device,
        "Safety risk dataset",
    )
    privacy_vectors = _compute_concept_vectors(
        args.base,
        privacy_anchor,
        privacy_loader,
        layer_names,
        device,
        "Privacy risk dataset",
    )
    layer_specs = [
        LayerDirections(
            name=layer,
            safe_vector=safety_vectors[layer],
            priv_vector=privacy_vectors[layer],
        )
        for layer in layer_names
    ]
    layer_map = {spec.name: spec for spec in layer_specs}
    base_vectors, layer_param_names = _prepare_base_vectors(
        Path(args.base), layer_specs, torch_dtype=None
    )
    baseline_reference_paths: Dict[str, Path] = {}
    defense_order: List[str] = []
    checkpoints: List[CheckpointSpec] = []
    next_auto_step = 0.0
    step_increment = 1.0
    if args.defense1:
        defense1_label = _derive_defense_label(args.defense1, "Defense1")
        defense_order.append(defense1_label)
        defense1_anchor_path = Path(args.defense1[-1])
        baseline_reference_paths[defense1_label] = defense1_anchor_path
        specs, next_auto_step = _build_simple_checkpoint_list(
            args.defense1,
            defense1_label,
            next_auto_step,
            step_increment,
            baseline="Base",
        )
        checkpoints.extend(specs)
    if args.defense2:
        defense2_label = _derive_defense_label(args.defense2, "Defense2")
        defense_order.append(defense2_label)
        specs, next_auto_step = _build_simple_checkpoint_list(
            args.defense2,
            defense2_label,
            next_auto_step,
            step_increment,
            baseline=defense1_label,
        )
        checkpoints.extend(specs)
    baseline_vectors: Dict[str, Dict[str, torch.Tensor]] = {"Base": base_vectors}
    if baseline_reference_paths:
        extra = _prepare_baseline_vectors(
            baseline_reference_paths, layer_param_names, torch_dtype=None
        )
        baseline_vectors.update(extra)
    args.output.mkdir(parents=True, exist_ok=True)
    records: List[dict] = [
        {
            "layer": layer_name,
            "defense": "Base",
            "step": 0.0,
            "checkpoint": str(Path(args.base)),
            "label": Path(args.base).name,
            "baseline": "Base",
            "alpha_safe": 0.0,
            "alpha_priv": 0.0,
            "delta_norm": 0.0,
            "angle_deg": 0.0,
            "alpha_safe_relative": 0.0,
            "alpha_priv_relative": 0.0,
            "delta_norm_relative": 0.0,
            "angle_deg_relative": 0.0,
        }
        for layer_name in layer_param_names
    ]
    records.extend(
        _aggregate_records(
            checkpoints,
            base_vectors,
            layer_param_names,
            layer_map,
            baseline_vectors,
            torch_dtype=None,
        )
    )
    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["layer", "step", "defense"]).reset_index(drop=True)
    csv_path = args.output / "dynamic_conflict_metrics.csv"
    df.to_csv(csv_path, index=False)
    for layer_name in layer_param_names:
        safe_name = layer_name.replace("/", "_").replace(".", "_")
        layer_dir = args.output / safe_name
        layer_dir.mkdir(parents=True, exist_ok=True)
        subset = df[df["layer"] == layer_name]
        _plot_alpha_curves(
            subset,
            layer_name,
            layer_dir / "alpha_vs_step.png",
            defense_order,
        )
        _plot_alpha_plane(
            subset,
            layer_name,
            layer_dir / "alpha_plane.png",
            defense_order,
        )
    print(f"Saved aggregated metrics to {csv_path}")


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = parse_args()
    run(args)


if __name__ == "__main__":  # pragma: no cover
    main()

