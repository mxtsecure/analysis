"""Analyse sequential defense interaction on safety representations only."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.datasets import load_dataset


MODEL_ORDER = ["base", "defense1", "defense2"]
MODEL_LABEL = {
    "base": "base LLM",
    "defense1": "base + safety defense",
    "defense2": "base + safety defense + privacy defense",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Path to base checkpoint")
    parser.add_argument("--defense1", required=True, help="Path to base+safety defense checkpoint")
    parser.add_argument("--defense2", required=True, help="Path to base+safety+privacy defense checkpoint")
    parser.add_argument("--safety-data", type=Path, required=True, help="JSONL containing safety prompts")
    parser.add_argument("--layer", type=int, default=None, help="Single layer index to analyze")
    parser.add_argument("--layers", default=None, help="Comma-separated layer indices (e.g. 7,12,16)")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-safety", type=int, default=0, help="Optional cap for safety prompts (0 = all)")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metrics and plots")
    parser.add_argument("--run-name", default="run", help="Sub-directory name under output-dir")
    return parser.parse_args()


def _resolve_dtype(label: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[label]


def _parse_layers(layer: int | None, layers: str | None) -> List[int]:
    if layers:
        parsed: List[int] = []
        for chunk in layers.split(","):
            chunk = chunk.strip()
            if chunk:
                parsed.append(int(chunk))
        if not parsed:
            raise ValueError("--layers is set but no valid layer index is provided")
        return sorted(set(parsed))
    if layer is None:
        raise ValueError("Please provide --layer or --layers")
    return [layer]


def _limit_dataset(dataset: Dataset, max_samples: int) -> Dataset:
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, list(range(max_samples)))


def _prepare_batch(batch: Mapping[str, object], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


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
            hidden = outputs.hidden_states[layer_idx + 1]
            last_positions = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
            row_idx = torch.arange(hidden.shape[0], device=device)
            vectors = hidden[row_idx, last_positions, :]
            chunks.append(vectors.detach().cpu())
    if not chunks:
        raise ValueError("No activation vectors collected for safety data")
    return torch.cat(chunks, dim=0)


def _mean_vectors(acts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {model_name: tensor.float().mean(dim=0) for model_name, tensor in acts.items()}


def _safe_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if torch.norm(a).item() == 0.0 or torch.norm(b).item() == 0.0:
        return float("nan")
    return float(F.cosine_similarity(a, b, dim=0).item())


def _compute_interaction_metrics(means: Dict[str, torch.Tensor]) -> Dict[str, float]:
    v_safety = means["defense1"] - means["base"]
    v_priv_on_safety = means["defense2"] - means["defense1"]
    v_total = means["defense2"] - means["base"]

    norm_v_safety = torch.norm(v_safety).item()
    norm_v_priv = torch.norm(v_priv_on_safety).item()
    norm_v_total = torch.norm(v_total).item()

    if norm_v_safety == 0.0:
        proj_ratio = float("nan")
        ortho_ratio = float("nan")
    else:
        unit_v_safety = v_safety / norm_v_safety
        proj_length = torch.dot(v_priv_on_safety, unit_v_safety).item()
        proj_ratio = proj_length / norm_v_safety
        ortho = v_priv_on_safety - proj_length * unit_v_safety
        ortho_ratio = torch.norm(ortho).item() / norm_v_safety

    return {
        "cos_v_priv_on_safety_vs_v_safety": _safe_cosine(v_priv_on_safety, v_safety),
        "norm_ratio_v_priv_on_safety_over_v_safety": float(norm_v_priv / norm_v_safety) if norm_v_safety > 0 else float("nan"),
        "projection_ratio_along_v_safety": float(proj_ratio),
        "orthogonal_ratio_to_v_safety": float(ortho_ratio),
        "cos_v_total_vs_v_safety": _safe_cosine(v_total, v_safety),
        "norms": {
            "v_safety": float(norm_v_safety),
            "v_priv_on_safety": float(norm_v_priv),
            "v_total": float(norm_v_total),
        },
    }


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


def _build_pca_points(acts: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    stacked = np.concatenate([acts[m].numpy() for m in MODEL_ORDER], axis=0)
    projected = _pca_2d(stacked)
    points: Dict[str, np.ndarray] = {}
    cursor = 0
    for model_name in MODEL_ORDER:
        n = acts[model_name].shape[0]
        points[model_name] = projected[cursor : cursor + n]
        cursor += n
    return points


def _save_plot(points: Dict[str, np.ndarray], metrics: Dict[str, float], path_prefix: Path, layer: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.4, 6.4))

    colors = {
        "base": "#0072B2",
        "defense1": "#E69F00",
        "defense2": "#009E73",
    }

    for model_name in MODEL_ORDER:
        coords = points[model_name]
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=14,
            alpha=0.16,
            color=colors[model_name],
            edgecolor="none",
            label=f"Safety samples ({MODEL_LABEL[model_name]})",
            zorder=2,
        )

    mean_traj = np.stack([points[m].mean(axis=0) for m in MODEL_ORDER], axis=0)

    ax.plot(
        mean_traj[:2, 0],
        mean_traj[:2, 1],
        color="#4D4D4D",
        linewidth=2.2,
        marker="o",
        markersize=6,
        label="Mean trajectory: base → safety defense",
        zorder=5,
    )
    ax.annotate("", xy=mean_traj[1], xytext=mean_traj[0], arrowprops=dict(arrowstyle="->", color="#4D4D4D", linewidth=1.8))

    ax.plot(
        mean_traj[1:, 0],
        mean_traj[1:, 1],
        color="#D55E00",
        linewidth=3.2,
        linestyle="--",
        marker="D",
        markersize=7,
        label="Mean trajectory: safety defense → privacy defense",
        zorder=6,
    )
    ax.annotate("", xy=mean_traj[2], xytext=mean_traj[1], arrowprops=dict(arrowstyle="->", color="#D55E00", linewidth=2.2))

    for i, model_name in enumerate(MODEL_ORDER):
        ax.text(mean_traj[i, 0], mean_traj[i, 1], f"  {MODEL_LABEL[model_name]}", fontsize=9, color="#222222", va="bottom")

    metrics_text = (
        f"cos(v_priv_on_safety, v_safety) = {metrics['cos_v_priv_on_safety_vs_v_safety']:.3f}\n"
        f"|v_priv_on_safety|/|v_safety| = {metrics['norm_ratio_v_priv_on_safety_over_v_safety']:.3f}\n"
        f"projection ratio along v_safety = {metrics['projection_ratio_along_v_safety']:.3f}\n"
        f"orthogonal ratio to v_safety = {metrics['orthogonal_ratio_to_v_safety']:.3f}\n"
        f"cos(v_total, v_safety) = {metrics['cos_v_total_vs_v_safety']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#666666", alpha=0.95),
    )

    ax.set_title(f"Sequential defense interaction on safety representation (layer {layer})", fontsize=13)
    ax.set_xlabel("PCA component 1", fontsize=10)
    ax.set_ylabel("PCA component 2", fontsize=10)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.legend(loc="best", fontsize=8, frameon=True)

    fig.tight_layout()
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_prefix.with_suffix(".png"), dpi=320)
    fig.savefig(path_prefix.with_suffix(".pdf"), dpi=320)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    selected_layers = _parse_layers(args.layer, args.layers)
    plot_layer = selected_layers[0]

    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    safety_dataset: Dataset = load_dataset(args.safety_data, tokenizer=tokenizer, max_length=args.max_length)
    safety_dataset = _limit_dataset(safety_dataset, args.max_safety)
    dataloader = DataLoader(safety_dataset, batch_size=args.batch_size, shuffle=False)

    models = {
        "base": AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype).to(device),
        "defense1": AutoModelForCausalLM.from_pretrained(args.defense1, torch_dtype=dtype).to(device),
        "defense2": AutoModelForCausalLM.from_pretrained(args.defense2, torch_dtype=dtype).to(device),
    }

    per_layer_activations: Dict[int, Dict[str, torch.Tensor]] = {layer: {} for layer in selected_layers}
    for model_name, model in models.items():
        for layer in selected_layers:
            per_layer_activations[layer][model_name] = _collect_last_token_activations(
                model,
                dataloader,
                layer_idx=layer,
                device=device,
            )
        model.cpu()
        del model

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()

    per_layer_metrics: Dict[str, Dict[str, float]] = {}
    for layer in selected_layers:
        means = _mean_vectors(per_layer_activations[layer])
        per_layer_metrics[str(layer)] = _compute_interaction_metrics(means)

    plot_points = _build_pca_points(per_layer_activations[plot_layer])
    _save_plot(
        points=plot_points,
        metrics=per_layer_metrics[str(plot_layer)],
        path_prefix=run_dir / "activation_trajectory_safety_only",
        layer=plot_layer,
    )

    payload = {
        "layers": selected_layers,
        "plot_layer": plot_layer,
        "datasets": {"safety": str(args.safety_data)},
        "num_samples": int(per_layer_activations[plot_layer]["base"].shape[0]),
        "metrics_by_layer": per_layer_metrics,
        "mean_trajectory_2d_plot_layer": {
            model_name: plot_points[model_name].mean(axis=0).tolist() for model_name in MODEL_ORDER
        },
    }

    json_path = run_dir / "activation_trajectory_safety_only.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics JSON: {json_path}")
    print(f"Saved plot PNG: {(run_dir / 'activation_trajectory_safety_only.png')}")
    print(f"Saved plot PDF: {(run_dir / 'activation_trajectory_safety_only.pdf')}")


if __name__ == "__main__":
    main()
