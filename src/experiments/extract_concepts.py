"""Script for Step 1: extracting purified safety/privacy concept vectors."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..analysis.concept_vectors import (
    ConceptVector,
    aggregate_difference,
    purify_concept_vector,
)
from ..data.datasets import build_dual_dataset
from ..models.activations import compute_activation_difference, forward_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base model name or path")
    parser.add_argument("--safety", required=True, help="Safety fine-tuned model")
    parser.add_argument("--privacy", required=True, help="Privacy fine-tuned model")
    parser.add_argument("--normal", required=True, help="Path to D_norm JSONL")
    parser.add_argument("--malicious", required=True, help="Path to D_mal JSONL")
    parser.add_argument("--privacy-data", required=True, help="Path to D_priv JSONL")
    parser.add_argument("--layer", required=True, help="Module name for activation capture")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--method",
        choices=["mean", "pca"],
        default="mean",
        help="Aggregation method for concept vectors",
    )
    return parser.parse_args()


def make_dataloaders(tokenizer, args) -> tuple[DataLoader, DataLoader, DataLoader]:
    datasets = build_dual_dataset(
        normal_path=args.normal,
        malicious_path=args.malicious,
        tokenizer=tokenizer,
        privacy_path=args.privacy_data,
    )
    if datasets.privacy is None:
        raise ValueError("Privacy dataset is required for concept extraction")
    normal_loader = DataLoader(datasets.normal, batch_size=args.batch_size)
    malicious_loader = DataLoader(datasets.malicious, batch_size=args.batch_size)
    privacy_loader = DataLoader(datasets.privacy, batch_size=args.batch_size)
    return normal_loader, malicious_loader, privacy_loader


def extract_concept_vectors(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    base_model = AutoModelForCausalLM.from_pretrained(args.base).to(device)
    safety_model = AutoModelForCausalLM.from_pretrained(args.safety).to(device)
    privacy_model = AutoModelForCausalLM.from_pretrained(args.privacy).to(device)

    normal_loader, malicious_loader, privacy_loader = make_dataloaders(tokenizer, args)

    base_norm = forward_dataset(base_model, normal_loader, args.layer, device, "Base D_norm")
    safety_norm = forward_dataset(
        safety_model, normal_loader, args.layer, device, "Safety D_norm"
    )
    base_mal = forward_dataset(base_model, malicious_loader, args.layer, device, "Base D_mal")
    safety_mal = forward_dataset(
        safety_model, malicious_loader, args.layer, device, "Safety D_mal"
    )

    norm_diff = compute_activation_difference(base_norm, safety_norm)
    mal_diff = compute_activation_difference(base_mal, safety_mal)

    v_norm = aggregate_difference(norm_diff, method=args.method)
    v_tilde_safety = aggregate_difference(mal_diff, method=args.method)
    v_safety = purify_concept_vector(v_tilde_safety, v_norm)

    base_priv = forward_dataset(base_model, privacy_loader, args.layer, device, "Base D_priv")
    privacy_priv = forward_dataset(
        privacy_model, privacy_loader, args.layer, device, "Privacy D_priv"
    )
    privacy_norm = forward_dataset(
        privacy_model, normal_loader, args.layer, device, "Privacy D_norm"
    )
    base_priv_norm = forward_dataset(
        base_model, normal_loader, args.layer, device, "Base D_norm (privacy)"
    )

    priv_diff = compute_activation_difference(base_priv, privacy_priv)
    priv_norm_diff = compute_activation_difference(base_priv_norm, privacy_norm)

    v_privacy_mixed = aggregate_difference(priv_diff, method=args.method)
    v_privacy_norm = aggregate_difference(priv_norm_diff, method=args.method)
    v_privacy = purify_concept_vector(v_privacy_mixed, v_privacy_norm)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(v_norm, args.output / "v_norm.pt")
    torch.save(v_safety, args.output / "v_safety.pt")
    torch.save(v_privacy, args.output / "v_privacy.pt")
    return {
        "v_norm": v_norm,
        "v_safety": v_safety,
        "v_privacy": v_privacy,
    }


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = parse_args()
    extract_concept_vectors(args)


if __name__ == "__main__":  # pragma: no cover
    main()
