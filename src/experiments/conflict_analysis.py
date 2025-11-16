"""Step 4: integrate concept and parameter metrics to explain conflicts."""
from __future__ import annotations

import argparse
import math
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
    parser.add_argument("--plot-path", type=Path)
    return parser.parse_args()


def load_concept(path: Path) -> ConceptVector:
    return torch.load(path, map_location="cpu")


def stack_concept_directions(*concepts: ConceptVector) -> torch.Tensor:
    """Stack concept directions for downstream linear algebra."""

    if not concepts:
        raise ValueError("At least one concept vector must be provided.")
    return torch.stack(
        [concept.direction.detach().to(dtype=torch.float32).cpu() for concept in concepts],
        dim=0,
    )


def compute_projection_basis(directions: torch.Tensor) -> torch.Tensor:
    """Compute a 2D projection basis using SVD."""

    if directions.ndim != 2 or directions.size(0) == 0:
        raise ValueError("Directions must be a non-empty 2D tensor.")
    # Use SVD to obtain an orthonormal basis spanning the supplied directions.
    # We operate entirely on CPU float tensors for numerical stability.
    try:
        _, _, vh = torch.linalg.svd(directions, full_matrices=False)
    except RuntimeError:
        # Fallback in the extremely rare case SVD fails due to degeneracy.
        directions = directions + 1e-12 * torch.randn_like(directions)
        _, _, vh = torch.linalg.svd(directions, full_matrices=False)
    basis = vh[:2]
    if basis.size(0) < 2:
        pad = torch.zeros((2 - basis.size(0), basis.size(1)), dtype=basis.dtype)
        basis = torch.cat([basis, pad], dim=0)
    return basis


def project_to_2d(directions: torch.Tensor) -> torch.Tensor:
    """Project the stacked concept directions into 2D coordinates."""

    basis = compute_projection_basis(directions)
    coords = directions @ basis.T
    return coords


def plot_concept_conflict(
    safety: ConceptVector,
    privacy: ConceptVector,
    similarity: float,
    save_path: Path,
) -> None:
    """Plot a 2D visualization of the concept conflict and save it."""

    import matplotlib.pyplot as plt

    directions = stack_concept_directions(safety, privacy)
    coords = project_to_2d(directions)
    coords_list = coords.tolist()
    labels = ["Safety", "Privacy"]
    colors = ["tab:blue", "tab:red"]

    fig, ax = plt.subplots(figsize=(6, 6))
    limit = max(float(coords.abs().max().item()), 1e-3)
    limit *= 1.2
    head_width = 0.03 * limit
    head_length = 0.05 * limit
    for (x, y), label, color in zip(coords_list, labels, colors):
        ax.arrow(0, 0, x, y, head_width=head_width, head_length=head_length, fc=color, ec=color)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), color=color)

    cos_val = max(-1.0, min(1.0, float(similarity)))
    angle_deg = math.degrees(math.acos(cos_val))
    annotation = f"cosθ = {cos_val:.3f}\nθ = {angle_deg:.1f}°"
    ax.text(0.05, 0.95, annotation, transform=ax.transAxes, va="top")

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("Concept conflict: safety vs privacy")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle="--", alpha=0.4)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:  # pragma: no cover - CLI
    args = parse_args()
    device = torch.device(args.device)
    base_model = AutoModelForCausalLM.from_pretrained(args.base).to(device)
    d1_model = AutoModelForCausalLM.from_pretrained(args.d1).to(device)
    d1d2_model = AutoModelForCausalLM.from_pretrained(args.d1d2).to(device)

    v_safety = load_concept(args.concept_safety)
    v_privacy = load_concept(args.concept_privacy)

    concept_similarity = concept_cos(v_safety, v_privacy)
    if args.plot_path:
        plot_concept_conflict(v_safety, v_privacy, concept_similarity, args.plot_path)
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
    if args.plot_path:
        print(f"Saved conflict plot to {args.plot_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
