"""Step 3: compute concept gradients and alignments with parameter updates."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..analysis.concept_vectors import ConceptVector
from ..analysis.gradients import compute_concept_gradient, cosine_similarity as grad_cos
from ..analysis.parameter_updates import (
    cosine_similarity as param_cos,
    parameter_update_vector,
)
from ..data.datasets import build_dual_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base model name or path")
    parser.add_argument("--d1", required=True, help="Model after applying defense D1")
    parser.add_argument("--d1d2", required=True, help="Model after applying D2 on top of D1")
    parser.add_argument("--normal", required=True, help="Path to D_norm JSONL")
    parser.add_argument("--malicious", required=True, help="Path to D_mal JSONL")
    parser.add_argument("--privacy-data", required=True, help="Path to D_priv JSONL")
    parser.add_argument("--concept-safety", type=Path, required=True)
    parser.add_argument("--concept-privacy", type=Path, required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_concept(path: Path) -> ConceptVector:
    return torch.load(path, map_location="cpu")


def main() -> None:  # pragma: no cover - CLI
    args = parse_args()
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    base_model = AutoModelForCausalLM.from_pretrained(args.base).to(device)
    d1_model = AutoModelForCausalLM.from_pretrained(args.d1).to(device)
    d1d2_model = AutoModelForCausalLM.from_pretrained(args.d1d2).to(device)

    datasets = build_dual_dataset(
        normal_path=args.normal,
        malicious_path=args.malicious,
        tokenizer=tokenizer,
        privacy_path=args.privacy_data,
    )
    malicious_loader = DataLoader(datasets.malicious, batch_size=args.batch_size)
    privacy_loader = DataLoader(datasets.privacy, batch_size=args.batch_size)

    v_safety: ConceptVector = load_concept(args.concept_safety)
    v_privacy: ConceptVector = load_concept(args.concept_privacy)

    g_safety = compute_concept_gradient(
        d1_model,
        malicious_loader,
        args.layer,
        v_safety,
        device,
        desc="Grad safety",
    )
    g_privacy = compute_concept_gradient(
        d1_model,
        privacy_loader,
        args.layer,
        v_privacy,
        device,
        desc="Grad privacy",
    )

    delta_theta_d1 = parameter_update_vector(base_model, d1_model)
    delta_theta_d2 = parameter_update_vector(d1_model, d1d2_model)

    align_d1 = param_cos(delta_theta_d1, g_safety)
    align_d2 = param_cos(delta_theta_d2, g_privacy)
    cross_align = param_cos(delta_theta_d1, delta_theta_d2)

    print(f"Alignment Δθ_D1 vs G_safety: {align_d1:.4f}")
    print(f"Alignment Δθ_D2 vs G_privacy: {align_d2:.4f}")
    print(f"Cross alignment Δθ_D1 vs Δθ_D2: {cross_align:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
