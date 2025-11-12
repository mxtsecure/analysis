"""Step 2: quantify overlap between safety and privacy concept vectors."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..analysis.concept_vectors import ConceptVector, cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safety", type=Path, required=True, help="Path to v_safety.pt")
    parser.add_argument("--privacy", type=Path, required=True, help="Path to v_privacy.pt")
    return parser.parse_args()


def load_concept(path: Path) -> ConceptVector:
    return torch.load(path, map_location="cpu")


def main() -> None:  # pragma: no cover - CLI
    args = parse_args()
    v_safety = load_concept(args.safety)
    v_privacy = load_concept(args.privacy)
    similarity = cosine_similarity(v_safety, v_privacy)
    print(f"Cosine similarity between safety and privacy concept vectors: {similarity:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
