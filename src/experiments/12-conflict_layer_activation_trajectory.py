"""Track activation trajectories on a conflict layer across sequential defenses."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Path to base LLM checkpoint")
    parser.add_argument("--defense1", required=True, help="Path to base+defense1 checkpoint")
    parser.add_argument("--defense2", required=True, help="Path to base+defense1+defense2 checkpoint")
    parser.add_argument("--input-data", type=Path, required=True, help="JSONL input set used by all models")
    parser.add_argument("--layer", type=int, required=True, help="Conflict layer index to analyze")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap on sample count (0 = all)")
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


def _prepare_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
    tensor_batch: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor_batch[key] = value.to(device)
    return tensor_batch


def _collect_layer_last_token(
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
            # hidden_states[0] is embeddings, hidden_states[1:] are transformer layers.
            target = hidden_states[layer_idx + 1]

            lengths = attention_mask.sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0)
            row_idx = torch.arange(target.shape[0], device=device)
            vectors = target[row_idx, lengths, :]
            chunks.append(vectors.detach().cpu())

    if not chunks:
        raise ValueError("No activation vectors were collected. Check dataset and dataloader settings.")
    return torch.cat(chunks, dim=0)


def _pca_2d(features: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("features must be 2D")
    centered = features - features.mean(axis=0, keepdims=True)
    if np.allclose(centered, 0):
        return np.zeros((features.shape[0], 2), dtype=np.float32)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    comps = vh[:2]
    projected = centered @ comps.T
    if projected.shape[1] < 2:
        projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])))
    return projected.astype(np.float32)


def _trajectory_metrics(base: torch.Tensor, d1: torch.Tensor, d2: torch.Tensor) -> Dict[str, object]:
    delta_01 = d1 - base
    delta_12 = d2 - d1
    delta_02 = d2 - base

    step_01 = torch.norm(delta_01, dim=1)
    step_12 = torch.norm(delta_12, dim=1)
    total = step_01 + step_12
    direct = torch.norm(delta_02, dim=1)
    curvature = total - direct

    return {
        "num_samples": int(base.shape[0]),
        "mean_step_base_to_defense1": float(step_01.mean().item()),
        "mean_step_defense1_to_defense2": float(step_12.mean().item()),
        "mean_direct_base_to_defense2": float(direct.mean().item()),
        "mean_path_length": float(total.mean().item()),
        "mean_curvature": float(curvature.mean().item()),
        "sample_metrics": [
            {
                "sample_index": i,
                "step_base_to_defense1": float(step_01[i].item()),
                "step_defense1_to_defense2": float(step_12[i].item()),
                "direct_base_to_defense2": float(direct[i].item()),
                "path_length": float(total[i].item()),
                "curvature": float(curvature[i].item()),
            }
            for i in range(base.shape[0])
        ],
    }


def _save_plot(points: Dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    colors = {"base": "#1f77b4", "defense1": "#ff7f0e", "defense2": "#2ca02c"}

    for name, coords in points.items():
        ax.scatter(coords[:, 0], coords[:, 1], s=18, alpha=0.55, label=name, color=colors[name])

    n = points["base"].shape[0]
    for i in range(n):
        x = [points["base"][i, 0], points["defense1"][i, 0], points["defense2"][i, 0]]
        y = [points["base"][i, 1], points["defense1"][i, 1], points["defense2"][i, 1]]
        ax.plot(x, y, color="gray", alpha=0.12, linewidth=0.8)

    mean_traj = np.stack([
        points["base"].mean(axis=0),
        points["defense1"].mean(axis=0),
        points["defense2"].mean(axis=0),
    ])
    ax.plot(
        mean_traj[:, 0],
        mean_traj[:, 1],
        color="black",
        linewidth=2.4,
        marker="o",
        markersize=6,
        label="mean trajectory",
    )

    ax.set_title("Conflict-layer activation trajectory (same input set)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    ax.grid(alpha=0.2, linestyle="--")
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

    dataset = load_dataset(args.input_data, tokenizer=tokenizer, max_length=args.max_length)
    if args.max_samples and args.max_samples > 0 and args.max_samples < len(dataset):
        dataset = Subset(dataset, list(range(args.max_samples)))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    models = {
        "base": AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype).to(device),
        "defense1": AutoModelForCausalLM.from_pretrained(args.defense1, torch_dtype=dtype).to(device),
        "defense2": AutoModelForCausalLM.from_pretrained(args.defense2, torch_dtype=dtype).to(device),
    }

    layer_acts: Dict[str, torch.Tensor] = {}
    for name, model in models.items():
        layer_acts[name] = _collect_layer_last_token(model, dataloader, layer_idx=args.layer, device=device)
        model.cpu()
        del model

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()

    metrics = _trajectory_metrics(layer_acts["base"], layer_acts["defense1"], layer_acts["defense2"])

    stacked = np.concatenate(
        [
            layer_acts["base"].numpy(),
            layer_acts["defense1"].numpy(),
            layer_acts["defense2"].numpy(),
        ],
        axis=0,
    )
    projected = _pca_2d(stacked)
    n = layer_acts["base"].shape[0]
    points = {
        "base": projected[:n],
        "defense1": projected[n : 2 * n],
        "defense2": projected[2 * n : 3 * n],
    }

    metrics["layer"] = args.layer
    metrics["input_data"] = str(args.input_data)
    metrics["points_2d"] = {
        name: coords.tolist() for name, coords in points.items()
    }
    metrics["mean_trajectory_2d"] = [
        points["base"].mean(axis=0).tolist(),
        points["defense1"].mean(axis=0).tolist(),
        points["defense2"].mean(axis=0).tolist(),
    ]

    with (run_dir / "activation_trajectory.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    _save_plot(points, run_dir / "activation_trajectory.png")
    print(f"Saved trajectory metrics to {run_dir / 'activation_trajectory.json'}")
    print(f"Saved trajectory plot to {run_dir / 'activation_trajectory.png'}")


if __name__ == "__main__":
    main()
