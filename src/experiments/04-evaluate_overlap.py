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
    results: Dict[str, float] = {}
    for layer in os.listdir(args.concepts_vector_path):
        if not layer.startswith("model_layers_"):
            continue
        print(f"Layer: {layer}")
        v_safety = load_concept(args.concepts_vector_path / layer / "v_safety.pt")
        v_privacy = load_concept(args.concepts_vector_path / layer / "v_privacy.pt")
        similarity = cosine_similarity(v_safety, v_privacy)
        results[layer] = round(similarity, 6)
        print(f"  {layer}: Cosine similarity = {similarity:.6f}")
    args.concepts_vector_overlap_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.concepts_vector_overlap_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":  # pragma: no cover
    main()
