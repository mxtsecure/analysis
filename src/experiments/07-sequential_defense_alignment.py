"""Analyse how sequential defenses interact across activations and weights."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.key_layers import collect_last_token_hidden_states
from data.datasets import RequestDataset, build_dual_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/finetune/Llama-3.2-1B-Instruct-tofu", help="Path to the base (pre-defense) checkpoint")
    parser.add_argument("--defense1", default="/data/xiangtao/projects/crossdefense/code/defense/safety/DPO/DPO_models/different/Llama-3.2-1B-Instruct-tofu-DPO", help="Path to the first defense checkpoint")
    parser.add_argument("--defense2", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/unlearn/Llama-3.2-1B-Instruct-tofu/Llama-3.2-1B-Instruct-tofu-NPO", help="Path to the second defense checkpoint")
    parser.add_argument("--normal", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/normal.jsonl", help="Path to D_norm JSONL")
    parser.add_argument("--malicious", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/safety.jsonl", help="Path to D_mal JSONL")
    parser.add_argument("--privacy-data", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/privacy.jsonl", help="Path to D_priv JSONL")
    parser.add_argument(
        "--layers",
        default="7",
        help="Comma separated list of layer indices to analyse (default: 10 evenly spaced)",
    )
    parser.add_argument("--max-malicious", type=int, default=512, help="Max samples drawn from D_mal")
    parser.add_argument("--max-privacy", type=int, default=512, help="Max samples drawn from D_priv")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/xiangtao/projects/crossdefense/code/analysis_results/07-sequential_defense_alignment/Llama-3.2-1B-Instruct-tofu"),
        help="Directory where the aggregated metrics are stored",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional override for the run sub-directory inside --output-dir",
    )
    return parser.parse_args()


def _limit_dataset(dataset: RequestDataset, max_samples: Optional[int], seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples]
    return Subset(dataset, indices.tolist())  # type: ignore[arg-type]


def _parse_layers(arg_value: Optional[str], num_layers: int) -> List[int]:
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


def _build_dataloaders(
    datasets: Mapping[str, RequestDataset],
    limits: Mapping[str, Optional[int]],
    *,
    seed: int,
    batch_size: int,
) -> Dict[str, DataLoader]:
    dataloaders: Dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        limited = _limit_dataset(dataset, limits.get(split), seed)
        dataloaders[split] = DataLoader(limited, batch_size=batch_size, shuffle=False)
    return dataloaders


def _compute_layer_means(states: Sequence[torch.Tensor], layers: Sequence[int]) -> Dict[int, torch.Tensor]:
    layer_means: Dict[int, torch.Tensor] = {}
    for layer in layers:
        tensor = states[layer].float()
        if tensor.ndim != 2:
            tensor = tensor.view(tensor.shape[0], -1)
        layer_means[layer] = tensor.mean(dim=0)
    return layer_means


def _difference_vector(
    baseline: Mapping[int, torch.Tensor],
    updated: Mapping[int, torch.Tensor],
    layers: Sequence[int],
) -> torch.Tensor:
    diffs: List[torch.Tensor] = []
    for layer in layers:
        if layer not in baseline or layer not in updated:
            raise KeyError(f"Layer {layer} missing from activation summaries")
        diffs.append((updated[layer] - baseline[layer]).reshape(-1))
    return torch.cat(diffs)


def _pairwise_cosine_table(vectors: Mapping[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    table: Dict[str, Dict[str, float]] = {}
    for left, right in itertools.product(vectors.keys(), repeat=2):
        if left not in table:
            table[left] = {}
        vec_l = vectors[left]
        vec_r = vectors[right]
        if vec_l.numel() == 0 or vec_r.numel() == 0:
            cosine = float("nan")
        else:
            cosine = F.cosine_similarity(vec_l, vec_r, dim=0).item()
        table[left][right] = cosine
    return table


def _extract_layer_parameter_deltas(
    state_a: Mapping[str, torch.Tensor],
    state_b: Mapping[str, torch.Tensor],
    layers: Sequence[int],
) -> Dict[int, torch.Tensor]:
    prefix = "model.layers."
    deltas: MutableMapping[int, List[torch.Tensor]] = {}
    layer_set = set(layers)
    for name, tensor_a in state_a.items():
        if not name.startswith(prefix) or name not in state_b:
            continue
        remainder = name[len(prefix) :]
        layer_str, _, _ = remainder.partition(".")
        if not layer_str.isdigit():
            continue
        layer_idx = int(layer_str)
        if layer_idx not in layer_set:
            continue
        tensor_b = state_b[name]
        if tensor_a.shape != tensor_b.shape:
            continue
        delta = (tensor_b - tensor_a).detach().float().reshape(-1)
        deltas.setdefault(layer_idx, []).append(delta)
    return {layer: torch.cat(chunks) for layer, chunks in deltas.items() if chunks}


def _save_heatmap(matrix: np.ndarray, labels: Sequence[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


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
    if datasets.privacy is None:
        raise ValueError("Privacy dataset is required for sequential defense analysis")

    dataloaders = _build_dataloaders(
        {
            "malicious": datasets.malicious,
            "privacy": datasets.privacy,
        },
        limits={"malicious": args.max_malicious, "privacy": args.max_privacy},
        seed=args.seed,
        batch_size=args.batch_size,
    )

    models = {
        "base": AutoModelForCausalLM.from_pretrained(args.base).to(device),
        "defense1": AutoModelForCausalLM.from_pretrained(args.defense1).to(device),
        "defense2": AutoModelForCausalLM.from_pretrained(args.defense2).to(device),
    }

    activation_states: Dict[str, Dict[str, List[torch.Tensor]]] = {model_name: {} for model_name in models}
    for model_name, model in models.items():
        for split, loader in dataloaders.items():
            activation_states[model_name][split] = collect_last_token_hidden_states(model, loader, device)

    num_layers = len(next(iter(activation_states.values()))["malicious"])
    selected_layers = _parse_layers(args.layers, num_layers)

    layer_means: Dict[str, Dict[str, Dict[int, torch.Tensor]]] = {}
    for model_name, splits in activation_states.items():
        layer_means[model_name] = {}
        for split, states in splits.items():
            layer_means[model_name][split] = _compute_layer_means(states, selected_layers)

    safety_shift = _difference_vector(
        layer_means["base"]["malicious"],
        layer_means["defense1"]["malicious"],
        selected_layers,
    )
    privacy_shift = _difference_vector(
        layer_means["defense1"]["privacy"],
        layer_means["defense2"]["privacy"],
        selected_layers,
    )
    interference_shift = _difference_vector(
        layer_means["defense1"]["malicious"],
        layer_means["defense2"]["malicious"],
        selected_layers,
    )

    shift_vectors = {
        "safety": safety_shift,
        "privacy": privacy_shift,
        "interference": interference_shift,
    }
    shift_norms = {name: torch.norm(vec).item() for name, vec in shift_vectors.items()}
    shift_cosines = _pairwise_cosine_table(shift_vectors)

    base_state = models["base"].state_dict()
    defense1_state = models["defense1"].state_dict()
    defense2_state = models["defense2"].state_dict()

    delta12 = _extract_layer_parameter_deltas(base_state, defense1_state, selected_layers)
    delta23 = _extract_layer_parameter_deltas(defense1_state, defense2_state, selected_layers)

    per_layer_param = []
    for layer in selected_layers:
        vec_a = delta12.get(layer)
        vec_b = delta23.get(layer)
        if vec_a is None or vec_b is None or vec_a.numel() == 0 or vec_b.numel() == 0:
            continue
        cosine = F.cosine_similarity(vec_a, vec_b, dim=0).item()
        per_layer_param.append(
            {
                "layer": layer,
                "delta12_norm": torch.norm(vec_a).item(),
                "delta23_norm": torch.norm(vec_b).item(),
                "cosine": cosine,
            }
        )

    run_name = args.run_name or "__".join(Path(path).name for path in [args.base, args.defense1, args.defense2])
    results_dir = args.output_dir / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "config": {
            "base": args.base,
            "defense1": args.defense1,
            "defense2": args.defense2,
            "normal": args.normal,
            "malicious": args.malicious,
            "privacy": args.privacy_data,
            "layers": selected_layers,
            "max_malicious": args.max_malicious,
            "max_privacy": args.max_privacy,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "activation_shift_norms": shift_norms,
        "activation_shift_cosines": shift_cosines,
        "parameter_alignment": per_layer_param,
    }

    metrics_path = results_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    csv_path = results_dir / "parameter_alignment.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write("layer,delta12_norm,delta23_norm,cosine\n")
        for row in per_layer_param:
            handle.write(
                f"{row['layer']},{row['delta12_norm']:.6f},{row['delta23_norm']:.6f},{row['cosine']:.6f}\n"
            )

    heatmap_path = results_dir / "activation_shift_cosines.png"
    labels = list(shift_vectors.keys())
    cosine_matrix = np.array([[shift_cosines[row][col] for col in labels] for row in labels])
    _save_heatmap(cosine_matrix, labels, heatmap_path)

    summary_lines = [
        "Sequential defense alignment:",
        f"  Safety shift norm (D_mal, defense1-base): {shift_norms['safety']:.4f}",
        f"  Privacy shift norm (D_priv, defense2-defense1): {shift_norms['privacy']:.4f}",
        f"  Interference shift norm (D_mal, defense2-defense1): {shift_norms['interference']:.4f}",
        f"  Cosine(safety, privacy): {shift_cosines['safety']['privacy']:.4f}",
        f"  Cosine(safety, interference): {shift_cosines['safety']['interference']:.4f}",
        f"  Cosine(privacy, interference): {shift_cosines['privacy']['interference']:.4f}",
    ]

    if per_layer_param:
        avg_cos = float(np.mean([row["cosine"] for row in per_layer_param]))
        summary_lines.append(f"  Mean parameter update cosine (delta12 vs delta23): {avg_cos:.4f}")
    else:
        summary_lines.append("  Parameter update cosines unavailable (layer mismatch)")

    print("\n".join(summary_lines))
    print(f"Results stored under {results_dir}")


if __name__ == "__main__":
    main()
