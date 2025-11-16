"""Step 4: integrate concept and parameter metrics to explain conflicts."""
from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
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
    from matplotlib.patches import FancyArrowPatch, Wedge

    directions = stack_concept_directions(safety, privacy)
    coords = project_to_2d(directions)
    coords_list = coords.tolist()
    labels = ["Safety", "Privacy"]
    colors = ["#2878b5", "#d14a61"]

    try:
        style_ctx = plt.style.context("seaborn-v0_8-whitegrid")
    except OSError:
        style_ctx = nullcontext()

    with style_ctx:
        fig, ax = plt.subplots(figsize=(7, 7))
        limit = max(float(coords.abs().max().item()), 1e-3)
        limit *= 1.25

        # soft radial backdrop to emphasize the angle between concepts
        angle_a = math.degrees(math.atan2(coords_list[0][1], coords_list[0][0]))
        angle_b = math.degrees(math.atan2(coords_list[1][1], coords_list[1][0]))
        start_angle, end_angle = sorted((angle_a, angle_b))
        wedge = Wedge(
            (0.0, 0.0),
            0.45 * limit,
            start_angle,
            end_angle,
            width=0.12 * limit,
            facecolor="#c2d5f2",
            edgecolor="none",
            alpha=0.35,
        )
        ax.add_patch(wedge)

        for (x, y), label, color in zip(coords_list, labels, colors):
            arrow = FancyArrowPatch(
                (0.0, 0.0),
                (x, y),
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=3.0,
                color=color,
                alpha=0.95,
            )
            ax.add_patch(arrow)
            ax.scatter([x], [y], color=color, s=80, zorder=5, edgecolors="white", linewidths=1.5)
            ax.annotate(
                label,
                (x, y),
                xytext=(6, 6),
                textcoords="offset points",
                color=color,
                fontsize=12,
                fontweight="bold",
            )

        cos_val = max(-1.0, min(1.0, float(similarity)))
        angle_deg = math.degrees(math.acos(cos_val))
        annotation = f"cosθ = {cos_val:.3f}\nθ = {angle_deg:.1f}°"
        ax.text(
            0.02,
            0.98,
            annotation,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
        )

        ax.set_xlabel("Projection component 1", fontsize=12)
        ax.set_ylabel("Projection component 2", fontsize=12)
        ax.set_title("Safety vs. Privacy Concept Geometry", fontsize=14, fontweight="bold")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", "box")
        ax.axhline(0, color="#888888", linewidth=1, alpha=0.6)
        ax.axvline(0, color="#888888", linewidth=1, alpha=0.6)
        ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
        ax.legend(labels, loc="upper right", frameon=False)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
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
