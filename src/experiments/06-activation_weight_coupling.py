"""Visualise how weight changes align with activation shifts across layers."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.key_layers import collect_last_token_hidden_states
from data.datasets import build_dual_dataset, RequestDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/finetune/Llama-3.2-1B-Instruct-tofu", help="Base (pre-defense) model path")
    parser.add_argument("--defense", default="/data/xiangtao/projects/crossdefense/code/defense/safety/DPO/DPO_models/different/Llama-3.2-1B-Instruct-tofu-DPO", help="Defense model path")
    parser.add_argument("--normal", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/normal.jsonl", help="Path to D_norm JSONL")
    parser.add_argument("--malicious", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/safety.jsonl", help="Path to D_mal JSONL")
    parser.add_argument("--privacy-data", help="Optional path to D_priv JSONL")
    parser.add_argument(
        "--split",
        choices=["normal", "malicious", "privacy"],
        default="malicious",
        help="Dataset split used to probe activations",
    )
    parser.add_argument(
        "--layers",
        default=None,
        help="Comma separated list of layer indices to plot (default: 10 evenly spaced)",
    )
    parser.add_argument("--max-samples", type=int, default=512, help="Samples used for activations")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=4096,
        help="Max parameter rows sampled per layer when estimating weight cosine",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("activation_weight_coupling.png"),
        help="Output path for the histogram grid",
    )
    return parser.parse_args()


def _select_dataset(datasets, split: str) -> RequestDataset:
    if split == "normal":
        return datasets.normal
    if split == "malicious":
        return datasets.malicious
    if datasets.privacy is None:
        raise ValueError("Privacy dataset requested but not provided")
    return datasets.privacy


def _limit_dataset(dataset: RequestDataset, max_samples: int, seed: int) -> Dataset:
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples]
    return Subset(dataset, indices.tolist())  # type: ignore[arg-type]


def _parse_layers(arg_value: str | None, num_layers: int) -> List[int]:
    if arg_value is None:
        if num_layers <= 10:
            return list(range(num_layers))
        positions = np.linspace(0, num_layers - 1, num=10)
        return sorted({int(round(p)) for p in positions})
    if arg_value.lower() == "all":
        return list(range(num_layers))
    layers: List[int] = []
    for chunk in arg_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = int(chunk)
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"Layer index {idx} is outside [0, {num_layers})")
        layers.append(idx)
    if not layers:
        raise ValueError("At least one layer must be selected")
    return sorted(set(layers))


def _collect_activation_cosine(
    base_states: Sequence[torch.Tensor], defense_states: Sequence[torch.Tensor]
) -> Dict[int, np.ndarray]:
    cosines: Dict[int, np.ndarray] = {}
    for layer_idx, (base, defense) in enumerate(zip(base_states, defense_states)):
        if base.shape != defense.shape:
            raise ValueError(f"Activation tensors mismatch at layer {layer_idx}")
        layer_cos = F.cosine_similarity(base, defense, dim=1)
        cosines[layer_idx] = layer_cos.cpu().numpy()
    return cosines


def _collect_weight_cosine(
    base_state: Dict[str, torch.Tensor],
    defense_state: Dict[str, torch.Tensor],
    *,
    max_rows: int,
    seed: int,
) -> Dict[int, np.ndarray]:
    prefix = "model.layers."
    rng = np.random.default_rng(seed)
    per_layer: Dict[int, List[np.ndarray]] = {}
    for name, base_tensor in base_state.items():
        if name not in defense_state or not name.startswith(prefix):
            continue
        remainder = name[len(prefix) :]
        layer_str, _, _ = remainder.partition(".")
        if not layer_str.isdigit():
            continue
        layer_idx = int(layer_str)
        other = defense_state[name]
        if base_tensor.ndim < 2 or other.ndim < 2:
            continue
        base_rows = base_tensor.float().view(base_tensor.shape[0], -1)
        defense_rows = other.float().view(other.shape[0], -1)
        if base_rows.shape != defense_rows.shape:
            continue
        cos = F.cosine_similarity(base_rows, defense_rows, dim=1).cpu().numpy()
        if max_rows and cos.shape[0] > max_rows:
            idx = rng.choice(cos.shape[0], size=max_rows, replace=False)
            cos = cos[idx]
        per_layer.setdefault(layer_idx, []).append(cos)
    return {layer: np.concatenate(chunks) for layer, chunks in per_layer.items() if chunks}


def _plot_histograms(
    layers: Sequence[int],
    activation_cos: Dict[int, np.ndarray],
    weight_cos: Dict[int, np.ndarray],
    output: Path,
) -> Path:
    if not layers:
        raise ValueError("No layers available for plotting")
    n_panels = len(layers)
    ncols = min(5, n_panels)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharey=True)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])
    bins = np.linspace(-1.0, 1.0, 40)
    for panel_idx, layer in enumerate(layers):
        row = panel_idx // ncols
        col = panel_idx % ncols
        ax = axes[row][col]
        act_values = activation_cos.get(layer)
        weight_values = weight_cos.get(layer)
        if act_values is None or weight_values is None:
            ax.axis("off")
            continue
        ax.hist(weight_values, bins=bins, density=True, alpha=0.65, color="#4C72B0", label="Weights")
        ax.hist(act_values, bins=bins, density=True, alpha=0.55, color="#DD8452", label="Activations")
        ax.set_title(f"Layer {layer}")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(bottom=0)
        if row == nrows - 1:
            ax.set_xlabel("Cosine similarity")
        if col == 0:
            ax.set_ylabel("Proportion")
        if panel_idx == 0:
            ax.legend()
    for extra in range(n_panels, nrows * ncols):
        row = extra // ncols
        col = extra % ncols
        axes[row][col].axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    datasets = build_dual_dataset(
        normal_path=args.normal,
        malicious_path=args.malicious,
        tokenizer=tokenizer,
        privacy_path=args.privacy_data,
    )
    dataset = _select_dataset(datasets, args.split)
    dataset = _limit_dataset(dataset, args.max_samples, args.seed)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    base_model = AutoModelForCausalLM.from_pretrained(args.base).to(device)
    defense_model = AutoModelForCausalLM.from_pretrained(args.defense).to(device)

    base_states = collect_last_token_hidden_states(base_model, dataloader, device)
    defense_states = collect_last_token_hidden_states(defense_model, dataloader, device)
    num_layers = len(base_states)

    selected_layers = _parse_layers(args.layers, num_layers)
    activation_cos = _collect_activation_cosine(base_states, defense_states)
    weight_cos = _collect_weight_cosine(
        base_model.state_dict(), defense_model.state_dict(), max_rows=args.max_rows, seed=args.seed
    )

    available_layers = [layer for layer in selected_layers if layer in activation_cos and layer in weight_cos]
    if not available_layers:
        raise ValueError("Selected layers do not have both activation and weight statistics")

    output_path = _plot_histograms(available_layers, activation_cos, weight_cos, args.output)
    act_means = [activation_cos[layer].mean() for layer in available_layers]
    weight_means = [weight_cos[layer].mean() for layer in available_layers]
    if len(act_means) > 1:
        pearson = np.corrcoef(act_means, weight_means)[0, 1]
        print(f"Pearson corr(mean cos): {pearson:.4f}")
    print(f"Saved histogram grid to {output_path}")


if __name__ == "__main__":
    main()
