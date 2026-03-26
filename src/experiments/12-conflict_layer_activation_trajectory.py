"""Track conflict-layer activation trajectories with safety and normal references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.datasets import RequestDataset


MODEL_ORDER = ["base", "defense1", "defense2"]
SPLIT_ORDER = ["safety", "normal"]
MODEL_LABEL = {
    "base": "Base",
    "defense1": "Defense1",
    "defense2": "Defense2",
}
SPLIT_LABEL = {
    "safety": "Safety prompts",
    "normal": "Normal prompts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Path to base checkpoint")
    parser.add_argument("--defense1", required=True, help="Path to base+defense1 checkpoint")
    parser.add_argument("--defense2", required=True, help="Path to base+defense1+defense2 checkpoint")
    parser.add_argument("--safety-data", type=Path, required=True, help="JSONL containing safety/malicious prompts")
    parser.add_argument("--normal-data", type=Path, required=True, help="JSONL containing normal benign prompts")
    parser.add_argument("--layer", type=int, required=True, help="Layer index to analyze")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-safety", type=int, default=0, help="Optional cap for safety prompts (0 = all)")
    parser.add_argument("--max-normal", type=int, default=0, help="Optional cap for normal prompts (0 = all)")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON metrics and plots")
    parser.add_argument("--run-name", default="run", help="Sub-directory name under output-dir")
    return parser.parse_args()


def _resolve_dtype(label: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[label]


def _limit_dataset(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples <= 0:
        return dataset
    if max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _prepare_batch(batch: Mapping[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
    tensor_batch: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor_batch[key] = value.to(device)
    return tensor_batch


def _collect_last_token_activations(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    layer_idx: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            tensor_batch = _prepare_batch(batch, device)
            attention_mask = tensor_batch["attention_mask"].long()
            outputs = model(**tensor_batch, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states
            target = hidden_states[layer_idx + 1]

            last_positions = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
            row_idx = torch.arange(target.shape[0], device=device)
            vectors = target[row_idx, last_positions, :]
            chunks.append(vectors.detach().cpu())

    if not chunks:
        raise ValueError("No activation vectors collected. Please check dataset and tokenizer settings.")
    return torch.cat(chunks, dim=0)


def _pca_2d(features: np.ndarray) -> np.ndarray:
    centered = features - features.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0):
        return np.zeros((features.shape[0], 2), dtype=np.float32)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    comps = vh[:2]
    projected = centered @ comps.T
    if projected.shape[1] < 2:
        projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
    return projected.astype(np.float32)


def _mean_vectors(acts: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
    means: Dict[str, Dict[str, torch.Tensor]] = {}
    for model_name, split_dict in acts.items():
        means[model_name] = {}
        for split_name, tensor in split_dict.items():
            means[model_name][split_name] = tensor.float().mean(dim=0)
    return means


def _compute_direction_metrics(means: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, object]:
    safety_direction = means["base"]["normal"] - means["base"]["safety"]

    defense1_shift = means["defense1"]["safety"] - means["base"]["safety"]
    defense2_shift = means["defense2"]["safety"] - means["defense1"]["safety"]
    full_shift = means["defense2"]["safety"] - means["base"]["safety"]

    def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        if torch.norm(a).item() == 0.0 or torch.norm(b).item() == 0.0:
            return float("nan")
        return float(F.cosine_similarity(a, b, dim=0).item())

    return {
        "reference": "base_normal_minus_base_safety",
        "reference_norm": float(torch.norm(safety_direction).item()),
        "cosine_alignment": {
            "base_to_defense1_on_safety": _safe_cosine(defense1_shift, safety_direction),
            "defense1_to_defense2_on_safety": _safe_cosine(defense2_shift, safety_direction),
            "base_to_defense2_on_safety": _safe_cosine(full_shift, safety_direction),
        },
        "shift_norms": {
            "base_to_defense1_on_safety": float(torch.norm(defense1_shift).item()),
            "defense1_to_defense2_on_safety": float(torch.norm(defense2_shift).item()),
            "base_to_defense2_on_safety": float(torch.norm(full_shift).item()),
        },
    }


def _build_pca_points(acts: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, np.ndarray]]:
    stacked: List[np.ndarray] = []
    keys: List[Tuple[str, str, int]] = []
    for model_name in MODEL_ORDER:
        for split_name in SPLIT_ORDER:
            arr = acts[model_name][split_name].numpy()
            stacked.append(arr)
            keys.append((model_name, split_name, arr.shape[0]))

    projected = _pca_2d(np.concatenate(stacked, axis=0))

    out: Dict[str, Dict[str, np.ndarray]] = {m: {} for m in MODEL_ORDER}
    cursor = 0
    for model_name, split_name, n in keys:
        out[model_name][split_name] = projected[cursor : cursor + n]
        cursor += n
    return out


def _save_plot(points: Dict[str, Dict[str, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    split_styles = {
        "safety": {"color": "#d62728", "marker": "o"},
        "normal": {"color": "#1f77b4", "marker": "^"},
    }

    for split_name in SPLIT_ORDER:
        style = split_styles[split_name]
        for model_name in MODEL_ORDER:
            coords = points[model_name][split_name]
            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                s=16,
                alpha=0.28,
                color=style["color"],
                marker=style["marker"],
                edgecolor="white",
                linewidths=0.2,
            )

        mean_traj = np.stack([points[m][split_name].mean(axis=0) for m in MODEL_ORDER], axis=0)
        ax.plot(
            mean_traj[:, 0],
            mean_traj[:, 1],
            color=style["color"],
            linewidth=2.0,
            marker=style["marker"],
            markersize=6,
            label=f"{SPLIT_LABEL[split_name]} mean trajectory",
            zorder=5,
        )

        for i in range(2):
            ax.annotate(
                "",
                xy=mean_traj[i + 1],
                xytext=mean_traj[i],
                arrowprops=dict(arrowstyle="->", color=style["color"], linewidth=1.4),
            )

    for model_name in MODEL_ORDER:
        safety_mean = points[model_name]["safety"].mean(axis=0)
        normal_mean = points[model_name]["normal"].mean(axis=0)
        ax.plot(
            [safety_mean[0], normal_mean[0]],
            [safety_mean[1], normal_mean[1]],
            linestyle="--",
            color="#555555",
            alpha=0.6,
            linewidth=1.0,
        )
        ax.text(
            normal_mean[0],
            normal_mean[1],
            MODEL_LABEL[model_name],
            fontsize=8,
            color="#2f2f2f",
            ha="left",
            va="bottom",
        )

    ax.set_title("Conflict-layer activation trajectories with normal reference", fontsize=12)
    ax.set_xlabel("PCA dimension 1", fontsize=10)
    ax.set_ylabel("PCA dimension 2", fontsize=10)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    safety_dataset: Dataset = RequestDataset(args.safety_data, tokenizer, max_words=args.max_length)
    normal_dataset: Dataset = RequestDataset(args.normal_data, tokenizer, max_words=args.max_length)
    safety_dataset = _limit_dataset(safety_dataset, args.max_safety)
    normal_dataset = _limit_dataset(normal_dataset, args.max_normal)

    dataloaders: Dict[str, DataLoader] = {
        "safety": DataLoader(safety_dataset, batch_size=args.batch_size, shuffle=False),
        "normal": DataLoader(normal_dataset, batch_size=args.batch_size, shuffle=False),
    }

    models = {
        "base": AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype).to(device),
        "defense1": AutoModelForCausalLM.from_pretrained(args.defense1, torch_dtype=dtype).to(device),
        "defense2": AutoModelForCausalLM.from_pretrained(args.defense2, torch_dtype=dtype).to(device),
    }

    activations: Dict[str, Dict[str, torch.Tensor]] = {m: {} for m in MODEL_ORDER}
    for model_name, model in models.items():
        for split_name, dataloader in dataloaders.items():
            activations[model_name][split_name] = _collect_last_token_activations(
                model,
                dataloader,
                layer_idx=args.layer,
                device=device,
            )
        model.cpu()
        del model

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()

    means = _mean_vectors(activations)
    direction_metrics = _compute_direction_metrics(means)
    points = _build_pca_points(activations)

    payload = {
        "layer": args.layer,
        "datasets": {
            "safety": str(args.safety_data),
            "normal": str(args.normal_data),
        },
        "num_samples": {
            split: int(next(iter(activations.values()))[split].shape[0]) for split in SPLIT_ORDER
        },
        "direction_metrics": direction_metrics,
        "mean_vectors": {
            model_name: {
                split_name: means[model_name][split_name].tolist()
                for split_name in SPLIT_ORDER
            }
            for model_name in MODEL_ORDER
        },
        "mean_trajectory_2d": {
            split_name: [
                points[model_name][split_name].mean(axis=0).tolist()
                for model_name in MODEL_ORDER
            ]
            for split_name in SPLIT_ORDER
        },
        "points_2d": {
            model_name: {
                split_name: points[model_name][split_name].tolist()
                for split_name in SPLIT_ORDER
            }
            for model_name in MODEL_ORDER
        },
    }

    json_path = run_dir / "activation_trajectory_with_normal.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    plot_path = run_dir / "activation_trajectory_with_normal.png"
    _save_plot(points, plot_path)

    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved trajectory plot: {plot_path}")


if __name__ == "__main__":
    main()
