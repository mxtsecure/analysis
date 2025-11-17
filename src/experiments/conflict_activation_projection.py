"""Project safety/privacy activations into 2D for conflict visualization."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.visualization import ConflictProjectionData, plot_conflict_projection
from data.datasets import build_dual_dataset
from models.activations import ActivationBatch, forward_dataset


def _normalize_layer_identifier(raw: str) -> str:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("Layer names must be non-empty")
    if stripped.isdigit():
        return f"model.layers.{stripped}"
    return stripped


def _parse_layer(arg_value: str | list[str]) -> str:
    if isinstance(arg_value, list):
        entries = arg_value
    else:
        entries = [arg_value]
    layers: list[str] = []
    for entry in entries:
        for part in entry.split(","):
            name = part.strip()
            if name:
                layers.append(_normalize_layer_identifier(name))
    if len(layers) != 1:
        raise ValueError("Exactly one layer must be provided for projection experiments")
    return layers[0]


def _select_last_token(batch: ActivationBatch) -> np.ndarray:
    hidden = batch.hidden_states
    if hidden.ndim == 2:
        return hidden.cpu().numpy()
    if hidden.ndim != 3:
        raise ValueError("Unexpected activation tensor shape")
    num_tokens = hidden.shape[1]
    if batch.attention_mask is None:
        indices = torch.full((hidden.shape[0],), num_tokens - 1, dtype=torch.long)
    else:
        mask = batch.attention_mask
        if mask.ndim != 2 or mask.shape[0] != hidden.shape[0]:
            raise ValueError("Attention mask must align with activation batch size")
        indices = mask.to(torch.int64).sum(dim=1) - 1
        indices = torch.clamp(indices, min=0)
    batch_indices = torch.arange(hidden.shape[0])
    vectors = hidden[batch_indices, indices, :]
    return vectors.cpu().numpy()


def _fit_projection(features: np.ndarray, method: str) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA

        projector = PCA(n_components=2)
        return projector.fit_transform(features)
    if method == "tsne":
        from sklearn.manifold import TSNE

        projector = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30)
        return projector.fit_transform(features)
    if method == "umap":
        from umap import UMAP

        projector = UMAP(n_components=2)
        return projector.fit_transform(features)
    raise ValueError(f"Unsupported projection method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/finetune/Llama-3.2-1B-Instruct-tofu", help="Base model name or path")
    parser.add_argument("--safety", default="/data/xiangtao/projects/crossdefense/code/defense/safety/DPO/DPO_models/different/Llama-3.2-1B-Instruct-tofu-DPO", help="Safety fine-tuned model")
    parser.add_argument("--privacy", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/unlearn/Llama-3.2-1B-Instruct-tofu/Llama-3.2-1B-Instruct-tofu-NPO", help="Privacy fine-tuned model")
    parser.add_argument("--normal", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/normal.jsonl", help="Path to D_norm JSONL")
    parser.add_argument("--malicious", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/safety.jsonl", help="Path to D_mal JSONL")
    parser.add_argument("--privacy-data", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/privacy.jsonl", help="Path to D_priv JSONL")
    parser.add_argument("--layer", dest="layer", action="append", default="16", help="Target layer for activation capture")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--projection", choices=["pca", "tsne", "umap"], default="pca", help="Projection method")
    parser.add_argument("--output", type=Path, default=Path("results/conflict_projection"), help="Output directory")
    args = parser.parse_args()
    if len(args.layer) == 1:
        args.layer = args.layer[0]
    return args


def conflict_projection(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    datasets = build_dual_dataset(
        normal_path=args.normal,
        malicious_path=args.malicious,
        tokenizer=tokenizer,
        privacy_path=args.privacy_data,
    )
    if datasets.privacy is None:
        raise ValueError("Privacy dataset is required for conflict visualization")
    malicious_loader = DataLoader(datasets.malicious, batch_size=args.batch_size)
    privacy_loader = DataLoader(datasets.privacy, batch_size=args.batch_size)

    base_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16).to(device)
    safety_model = AutoModelForCausalLM.from_pretrained(args.safety, torch_dtype=torch.float16).to(device)
    privacy_model = AutoModelForCausalLM.from_pretrained(args.privacy, torch_dtype=torch.float16).to(device)

    layer = _parse_layer(args.layer)

    def capture(model, dataloader, desc: str) -> np.ndarray:
        activations = forward_dataset(model, dataloader, layer, device, desc)
        if isinstance(activations, dict):
            activation_batch = activations[layer]
        else:
            activation_batch = activations
        return _select_last_token(activation_batch)

    base_safety = capture(base_model, malicious_loader, "Base D_mal")
    defense_safety = capture(safety_model, malicious_loader, "Safety model D_mal")
    base_privacy = capture(base_model, privacy_loader, "Base D_priv")
    defense_privacy = capture(privacy_model, privacy_loader, "Privacy model D_priv")

    groups = {
        "base_safety": base_safety,
        "defense_safety": defense_safety,
        "base_privacy": base_privacy,
        "defense_privacy": defense_privacy,
    }
    feature_matrix = np.concatenate(list(groups.values()), axis=0)
    projected = _fit_projection(feature_matrix, args.projection)

    split_points: dict[str, np.ndarray] = {}
    start = 0
    for name, tensor in groups.items():
        length = tensor.shape[0]
        split_points[name] = projected[start : start + length]
        start += length

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output / "projection_coords.npz",
        **split_points,
        layer=layer,
        method=args.projection,
    )

    projection_data = ConflictProjectionData(
        base_safety=split_points["base_safety"],
        defense_safety=split_points["defense_safety"],
        base_privacy=split_points["base_privacy"],
        defense_privacy=split_points["defense_privacy"],
    )
    title = f"Layer {layer} — {args.projection.upper()} projection"
    plot_conflict_projection(
        projection_data,
        args.output / "conflict_projection.png",
        title=title,
    )


def main() -> None:
    args = parse_args()
    conflict_projection(args)


if __name__ == "__main__":
    main()
