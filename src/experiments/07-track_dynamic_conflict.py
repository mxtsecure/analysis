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
from transformers import AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def _normalize_layer_name(raw: str) -> str:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("Layer identifiers must be non-empty")
    if stripped.isdigit():
        return f"model.layers.{stripped}"
    return stripped


def _load_direction_vector(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Direction file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path)
        tensor = torch.from_numpy(array)
    elif suffix in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.Tensor):
            tensor = obj
        elif hasattr(obj, "direction"):
            tensor = getattr(obj, "direction")
        elif isinstance(obj, Mapping) and "direction" in obj:
            tensor = obj["direction"]
        else:
            raise ValueError(f"Unsupported tensor container stored in {path}")
    else:
        raise ValueError(f"Unsupported direction file extension: {path.suffix}")
    tensor = tensor.detach().cpu().to(torch.float64).flatten()
    norm = torch.norm(tensor)
    if norm.item() == 0:
        raise ValueError(f"Direction vector from {path} is zero")
    return tensor / norm


def _parse_layer_specs(values: Sequence[str]) -> List[LayerDirections]:
    if not values:
        raise ValueError("At least one --layer-spec entry is required")
    specs: List[LayerDirections] = []
    for raw in values:
        parts = [part.strip() for part in raw.split(":") if part.strip()]
        if len(parts) != 3:
            raise ValueError(
                "--layer-spec expects entries formatted as 'layer:safe_path:priv_path'"
            )
        layer = _normalize_layer_name(parts[0])
        safe_vector = _load_direction_vector(Path(parts[1]))
        priv_vector = _load_direction_vector(Path(parts[2]))
        specs.append(
            LayerDirections(
                name=layer,
                safe_vector=safe_vector,
                priv_vector=priv_vector,
            )
        )
    return specs


def _parse_checkpoint_list(
    values: Sequence[str],
    defense: str,
    next_auto_step: float,
    step_increment: float,
) -> tuple[List[CheckpointSpec], float]:
    specs: List[CheckpointSpec] = []
    current_auto_step = next_auto_step
    for raw in values:
        entry = raw.strip()
        if not entry:
            continue
        if "=" in entry:
            step_str, path_str = entry.split("=", 1)
            try:
                step = float(step_str)
            except ValueError as exc:
                raise ValueError(f"Invalid step value '{step_str}' in '{entry}'") from exc
            current_auto_step = step + step_increment
        else:
            path_str = entry
            step = current_auto_step
            current_auto_step += step_increment
        path = Path(path_str)
        label = path.name
        specs.append(CheckpointSpec(path=path, step=step, label=label, defense=defense))
    return specs, current_auto_step


def _select_layer_parameters(state_dict_keys: Iterable[str], layer: str) -> List[str]:
    prefix = f"{layer}."
    selected = [name for name in state_dict_keys if name == layer or name.startswith(prefix)]
    selected.sort()
    if not selected:
        raise ValueError(f"Layer '{layer}' was not found in the checkpoint state dict")
    return selected


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
            label=f"{defense} α_safe",
            color=color,
            linestyle="-",
            marker="o",
        )
        ax.plot(
            subset["step"],
            subset["alpha_priv"],
            label=f"{defense} α_priv",
            color=color,
            linestyle="--",
            marker="o",
            alpha=0.8,
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
            label=f"{defense} trajectory",
        )
        for _, row in subset.iterrows():
            ax.annotate(
                f"{row['step']:.0f}",
                (row["alpha_safe"], row["alpha_priv"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
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
    parser.add_argument("--base-step", type=float, default=0.0, help="Step assigned to the base checkpoint")
    parser.add_argument(
        "--defense1",
        nargs="+",
        default=[],
        help="Sequence of checkpoints for defense 1 (format: step=path).",
    )
    parser.add_argument(
        "--defense2",
        nargs="+",
        default=[],
        help="Sequence of checkpoints for defense 2 (format: step=path).",
    )
    parser.add_argument("--defense1-name", default="Defense1", help="Label for the first defense trajectory")
    parser.add_argument("--defense2-name", default="Defense2", help="Label for the second defense trajectory")
    parser.add_argument(
        "--layer-spec",
        action="append",
        required=True,
        help="Per-layer direction spec formatted as layer:safe_path:priv_path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/xiangtao/projects/crossdefense/code/analysis_results/07-dynamic_conflict"),
        help="Directory where CSV and plots will be stored",
    )
    parser.add_argument(
        "--step-increment",
        type=float,
        default=1.0,
        help=(
            "Increment applied to automatically assigned step values when explicit"
            " steps are omitted"
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "float32", "bfloat16", "auto"],
        default="auto",
        help="Optional torch dtype override when loading checkpoints",
    )
    return parser.parse_args()


def _resolve_torch_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


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


def _aggregate_records(
    checkpoints: Sequence[CheckpointSpec],
    base_vectors: Mapping[str, torch.Tensor],
    layer_param_names: Mapping[str, Sequence[str]],
    layer_specs: Mapping[str, LayerDirections],
    torch_dtype: torch.dtype | None,
) -> List[dict]:
    include_keys = sorted({name for names in layer_param_names.values() for name in names})
    records: List[dict] = []
    for spec in checkpoints:
        state = _load_checkpoint_state(spec.path, include=include_keys, torch_dtype=torch_dtype)
        for layer_name, names in layer_param_names.items():
            delta = _compute_delta(base_vectors[layer_name], state, names)
            layer_dir = layer_specs[layer_name]
            alpha_safe, alpha_priv, delta_norm, angle = _compute_metrics(
                delta, layer_dir.safe_vector, layer_dir.priv_vector
            )
            records.append(
                {
                    "layer": layer_name,
                    "defense": spec.defense,
                    "step": spec.step,
                    "checkpoint": str(spec.path),
                    "label": spec.label,
                    "alpha_safe": alpha_safe,
                    "alpha_priv": alpha_priv,
                    "delta_norm": delta_norm,
                    "angle_deg": angle,
                }
            )
        del state
    return records


def run(args: argparse.Namespace) -> None:
    torch_dtype = _resolve_torch_dtype(args.torch_dtype)
    layer_specs = _parse_layer_specs(args.layer_spec)
    layer_map = {spec.name: spec for spec in layer_specs}
    base_vectors, layer_param_names = _prepare_base_vectors(
        Path(args.base), layer_specs, torch_dtype
    )
    if not args.defense1 and not args.defense2:
        raise ValueError("At least one defense checkpoint list must be provided")
    defense_order: List[str] = []
    checkpoints: List[CheckpointSpec] = []
    next_auto_step = float(args.base_step) + float(args.step_increment)
    if args.defense1:
        defense_order.append(args.defense1_name)
        specs, next_auto_step = _parse_checkpoint_list(
            args.defense1, args.defense1_name, next_auto_step, float(args.step_increment)
        )
        checkpoints.extend(specs)
    if args.defense2:
        defense_order.append(args.defense2_name)
        specs, next_auto_step = _parse_checkpoint_list(
            args.defense2, args.defense2_name, next_auto_step, float(args.step_increment)
        )
        checkpoints.extend(specs)
    args.output.mkdir(parents=True, exist_ok=True)
    records: List[dict] = [
        {
            "layer": layer_name,
            "defense": "Base",
            "step": float(args.base_step),
            "checkpoint": str(Path(args.base)),
            "label": Path(args.base).name,
            "alpha_safe": 0.0,
            "alpha_priv": 0.0,
            "delta_norm": 0.0,
            "angle_deg": 0.0,
        }
        for layer_name in layer_param_names
    ]
    records.extend(
        _aggregate_records(
            checkpoints,
            base_vectors,
            layer_param_names,
            layer_map,
            torch_dtype,
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

