"""Analyze projection ratios between safety and privacy activation shifts across layers."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.datasets import build_dual_dataset
from models.activations import ActivationBatch, forward_dataset, _select_final_token


DEFAULT_LAYER_STRING = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
DEFAULT_OUTPUT_ROOT = Path("analysis_results/08-activation_projection_ratio")


def _normalize_layer_identifier(raw: str) -> str:
    stripped = raw.strip()
    if not stripped:
        raise ValueError("Layer names must be non-empty")
    if stripped.isdigit():
        return f"model.layers.{stripped}"
    return stripped


def _parse_layers(arg_value: str | Sequence[str]) -> List[str]:
    if isinstance(arg_value, list):
        entries = arg_value
    else:
        entries = [arg_value]
    layers: List[str] = []
    for entry in entries:
        for part in entry.split(","):
            name = part.strip()
            if name:
                layers.append(_normalize_layer_identifier(name))
    if not layers:
        raise ValueError("At least one layer must be provided for projection experiments")
    return layers


def _select_final_vectors(batch: ActivationBatch) -> torch.Tensor:
    return _select_final_token(batch.hidden_states, batch.attention_mask)


def _mean_difference(base: ActivationBatch, finetuned: ActivationBatch) -> torch.Tensor:
    base_vec = _select_final_vectors(base)
    finetuned_vec = _select_final_vectors(finetuned)
    if base_vec.shape != finetuned_vec.shape:
        raise ValueError("Activation tensors must share the same shape for differencing")
    return finetuned_vec.mean(dim=0) - base_vec.mean(dim=0)


def _compute_metrics(
    safety_diff: torch.Tensor,
    privacy_diff: torch.Tensor,
    *,
    small_ratio_threshold: float,
) -> dict:
    eps = 1e-12
    safety_norm = safety_diff.norm().item()
    privacy_norm = privacy_diff.norm().item()

    dot = torch.dot(safety_diff, privacy_diff).item()
    cosine = dot / max(eps, safety_norm * privacy_norm)

    proj_priv_on_safety = dot / max(eps, safety_norm)
    priv_ratio = proj_priv_on_safety / max(eps, privacy_norm)

    proj_safety_on_privacy = dot / max(eps, privacy_norm)
    safety_ratio = proj_safety_on_privacy / max(eps, safety_norm)

    conflict = cosine < 0 and abs(proj_priv_on_safety) < small_ratio_threshold * max(eps, privacy_norm)

    return {
        "cosine_similarity": cosine,
        "projection_priv_on_safety": proj_priv_on_safety,
        "projection_priv_ratio": priv_ratio,
        "projection_safety_on_privacy": proj_safety_on_privacy,
        "projection_safety_ratio": safety_ratio,
        "safety_norm": safety_norm,
        "privacy_norm": privacy_norm,
        "conflict_flag": conflict,
    }


def _save_table(rows: List[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    json_path = output_dir / "metrics.json"

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def _plot_metrics(rows: List[dict], output_dir: Path) -> None:
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = [row["layer"] for row in rows]
    cosine = [row["cosine_similarity"] for row in rows]
    proj_priv = [row["projection_priv_on_safety"] for row in rows]
    proj_priv_ratio = [row["projection_priv_ratio"] for row in rows]
    proj_safety = [row["projection_safety_on_privacy"] for row in rows]
    proj_safety_ratio = [row["projection_safety_ratio"] for row in rows]

    plt.style.use("default")

    fig, axes = plt.subplots(3, 2, figsize=(10, 9))
    axes = axes.flatten()

    axes[0].bar(layers, cosine, color="#4C72B0")
    axes[0].set_title("Cosine Similarity")
    axes[0].set_ylabel("cos")

    axes[1].bar(layers, proj_priv, color="#55A868")
    axes[1].set_title("Privacy projection on Safety")
    axes[1].set_ylabel("scalar projection")

    axes[2].bar(layers, proj_priv_ratio, color="#C44E52")
    axes[2].set_title("Privacy projection ratio")
    axes[2].set_ylabel("|proj| / |privacy|")

    axes[3].bar(layers, proj_safety, color="#8172B3")
    axes[3].set_title("Safety projection on Privacy")
    axes[3].set_ylabel("scalar projection")

    axes[4].bar(layers, proj_safety_ratio, color="#CCB974")
    axes[4].set_title("Safety projection ratio")
    axes[4].set_ylabel("|proj| / |safety|")

    axes[5].scatter(layers, cosine, color="#4C72B0")
    axes[5].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[5].set_title("Cosine Trend")
    axes[5].set_ylabel("cos")

    for ax in axes:
        ax.set_xlabel("Layer")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "projections.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/finetune/Llama-3.2-1B-Instruct-tofu", help="Base model name or path")
    parser.add_argument("--safety", default="/data/xiangtao/projects/crossdefense/code/defense/safety/DPO/DPO_models/different/Llama-3.2-1B-Instruct-tofu-DPO", help="Safety fine-tuned model")
    parser.add_argument("--privacy", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/unlearn/Llama-3.2-1B-Instruct-tofu/Llama-3.2-1B-Instruct-tofu-NPO", help="Privacy fine-tuned model")
    parser.add_argument("--normal", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/normal.jsonl", help="Path to D_norm JSONL")
    parser.add_argument("--malicious", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/safety.jsonl", help="Path to D_mal JSONL")
    parser.add_argument("--privacy-data", required=True, dest="privacy_data", help="Path to D_priv JSONL")
    parser.add_argument("--layer", dest="layer", action="append", default=DEFAULT_LAYER_STRING, help="Target layer(s) for activation capture")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional limit on dataloader batches")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for analysis outputs")
    parser.add_argument("--run-name", type=str, default="default", help="Run identifier used to create a subdirectory")
    parser.add_argument("--small-ratio-threshold", type=float, default=0.1, help="Threshold for marking conflicting directions")
    parser.add_argument("--no-plot", action="store_true", help="Disable plot generation")
    return parser.parse_args()


def activation_projection_ratio(args: argparse.Namespace) -> None:
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    datasets = build_dual_dataset(
        normal_path=args.normal,
        malicious_path=args.malicious,
        tokenizer=tokenizer,
        privacy_path=args.privacy_data,
    )
    if datasets.privacy is None:
        raise ValueError("Privacy dataset is required for projection ratio analysis")

    malicious_loader = DataLoader(datasets.malicious, batch_size=args.batch_size)
    privacy_loader = DataLoader(datasets.privacy, batch_size=args.batch_size)

    base_model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float16).to(device)
    safety_model = AutoModelForCausalLM.from_pretrained(args.safety, torch_dtype=torch.float16).to(device)
    privacy_model = AutoModelForCausalLM.from_pretrained(args.privacy, torch_dtype=torch.float16).to(device)

    layers = _parse_layers(args.layer)

    def capture(model, dataloader, desc: str) -> Dict[str, ActivationBatch]:
        activations = forward_dataset(
            model,
            dataloader,
            layers,
            device,
            desc,
            max_batches=args.max_batches,
        )
        if not isinstance(activations, dict):
            activations = {layers[0]: activations}
        return activations

    base_mal = capture(base_model, malicious_loader, "Base D_mal")
    safety_mal = capture(safety_model, malicious_loader, "Safety D_mal")
    base_priv = capture(base_model, privacy_loader, "Base D_priv")
    privacy_priv = capture(privacy_model, privacy_loader, "Privacy D_priv")

    rows: List[dict] = []
    for layer in layers:
        safety_diff = _mean_difference(base_mal[layer], safety_mal[layer])
        privacy_diff = _mean_difference(base_priv[layer], privacy_priv[layer])

        metrics = _compute_metrics(
            safety_diff,
            privacy_diff,
            small_ratio_threshold=args.small_ratio_threshold,
        )
        rows.append({"layer": layer, **metrics})

    output_dir = args.output / args.run_name
    _save_table(rows, output_dir)
    if not args.no_plot:
        _plot_metrics(rows, output_dir)



def main() -> None:
    args = parse_args()
    activation_projection_ratio(args)


if __name__ == "__main__":
    main()
