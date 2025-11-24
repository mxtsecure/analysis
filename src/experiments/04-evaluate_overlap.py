"""Step 2: quantify overlap between safety and privacy concept vectors."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import json
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.concept_vectors import ConceptVector, cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts_vector_path", type=Path, default=Path("/data/xiangtao/projects/crossdefense/code/analysis/results/01-concepts_vector/Llama-3.2-1B-Instruct-tofu/accurate"), help="Path to concepts vector directory")
    parser.add_argument("--concepts_vector_overlap_path", type=Path, default=Path("/data/xiangtao/projects/crossdefense/code/analysis/results/02-concepts_vector_overlap/Llama-3.2-1B-Instruct-tofu/accurate/result.json"), help="Path to concepts vector directory")
    return parser.parse_args()


def load_concept(path: Path) -> ConceptVector:
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:  # pragma: no cover - CLI
    args = parse_args()
    results: Dict[str, Dict[str, float]] = {}
    for layer in os.listdir(args.concepts_vector_path):
        if not layer.startswith("model_layers_"):
            continue
        print(f"Layer: {layer}")
        v_safety = load_concept(args.concepts_vector_path / layer / "v_safety.pt")
        v_privacy = load_concept(args.concepts_vector_path / layer / "v_privacy.pt")

        safety_norm = v_safety.direction.norm(p=2).item()
        privacy_norm = v_privacy.direction.norm(p=2).item()
        raw_overlap = torch.dot(v_safety.direction, v_privacy.direction).item()
        similarity = cosine_similarity(v_safety, v_privacy)
        magnitude_weighted_overlap = raw_overlap / (max(safety_norm, privacy_norm) + 1e-12)

        results[layer] = {
            "cosine_similarity": round(similarity, 6),
            "safety_norm": round(safety_norm, 6),
            "privacy_norm": round(privacy_norm, 6),
            "magnitude_weighted_overlap": round(magnitude_weighted_overlap, 6),
        }

        print(
            "  "
            f"safety_norm={safety_norm:.6f}, privacy_norm={privacy_norm:.6f}, "
            f"cosine_similarity={similarity:.6f}, "
            f"magnitude_weighted_overlap={magnitude_weighted_overlap:.6f}"
        )
    args.concepts_vector_overlap_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.concepts_vector_overlap_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":  # pragma: no cover
    main()
