"""Step 4: integrate concept and parameter metrics to explain conflicts."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from ..analysis.concept_vectors import ConceptVector, cosine_similarity as concept_cos
from ..analysis.parameter_updates import cosine_similarity as param_cos, parameter_update_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--d1", required=True)
    parser.add_argument("--d1d2", required=True)
    parser.add_argument("--concept-safety", type=Path, required=True)
    parser.add_argument("--concept-privacy", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_concept(path: Path) -> ConceptVector:
    return torch.load(path, map_location="cpu")


def main() -> None:  # pragma: no cover - CLI
    args = parse_args()
    device = torch.device(args.device)
    base_model = AutoModelForCausalLM.from_pretrained(args.base).to(device)
    d1_model = AutoModelForCausalLM.from_pretrained(args.d1).to(device)
    d1d2_model = AutoModelForCausalLM.from_pretrained(args.d1d2).to(device)

    v_safety = load_concept(args.concept_safety)
    v_privacy = load_concept(args.concept_privacy)

    concept_similarity = concept_cos(v_safety, v_privacy)
    delta_theta_d1 = parameter_update_vector(base_model, d1_model)
    delta_theta_d2 = parameter_update_vector(d1_model, d1d2_model)
    parameter_similarity = param_cos(delta_theta_d1, delta_theta_d2)

    print("=== Conflict analysis summary ===")
    print(f"Concept cosine similarity (safety vs privacy): {concept_similarity:.4f}")
    print(f"Parameter cosine similarity (Δθ_D1 vs Δθ_D2): {parameter_similarity:.4f}")
    if parameter_similarity < 0:
        print("Negative alignment indicates D2 may undo D1 along overlapping concepts.")
    else:
        print("Positive alignment indicates D2 likely cooperates with D1.")


if __name__ == "__main__":  # pragma: no cover
    main()
