"""Step 4: integrate concept and parameter metrics to explain conflicts."""
from __future__ import annotations

import argparse
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.concept_vectors import ConceptVector, cosine_similarity as concept_cos
from analysis.parameter_updates import cosine_similarity as param_cos, parameter_update_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/finetune/Llama-3.2-1B-Instruct-tofu")
    parser.add_argument("--d1", default="/data/xiangtao/projects/crossdefense/code/defense/safety/DPO/DPO_models/different/Llama-3.2-1B-Instruct-tofu-DPO")
    parser.add_argument("--d1d2", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/unlearn/Llama-3.2-1B-Instruct-tofu/Llama-3.2-1B-Instruct-tofu-DPO-NPO")
    parser.add_argument("--concept-safety", type=Path, default="/data/xiangtao/projects/crossdefense/code/analysis_results/01-concepts_vector/Llama-3.2-1B-Instruct-tofu/fast/model_layers_7/v_safety.pt")
    parser.add_argument("--concept-privacy", type=Path, default="/data/xiangtao/projects/crossdefense/code/analysis_results/01-concepts_vector/Llama-3.2-1B-Instruct-tofu/fast/model_layers_7/v_privacy.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--plot-path", type=Path, default="/data/xiangtao/projects/crossdefense/code/analysis_results/03-conflict/Llama-3.2-1B-Instruct-tofu/result.png")
    return parser.parse_args()


def load_concept(path: Path) -> ConceptVector:
    return torch.load(path, map_location="cpu", weights_only=False)


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


def _coerce_sample_tensor(samples, name: str | None = None) -> torch.Tensor | None:
    """Convert user supplied samples to a 2D float tensor if provided."""

    if samples is None:
        return None
    tensor = torch.as_tensor(samples, dtype=torch.float32)
    if tensor.requires_grad:
        tensor = tensor.detach()
    tensor = tensor.cpu()
    if tensor.ndim != 2:
        raise ValueError(
            f"{name or 'Samples'} must be a 2D tensor/array, got shape {tuple(tensor.shape)}"
        )
    return tensor


def plot_concept_conflict(
    safety: ConceptVector,
    privacy: ConceptVector,
    similarity: float,
    save_path: Path,
    baseline_samples: torch.Tensor | np.ndarray | None = None,
    safety_samples: torch.Tensor | np.ndarray | None = None,
    privacy_samples: torch.Tensor | np.ndarray | None = None,
) -> None:
    """Plot either the concept comparison or neuron activation shifts.

    Args:
        safety: Learned safety concept vector.
        privacy: Learned privacy concept vector.
        similarity: Cosine similarity between the two concepts.
        save_path: Destination for the saved figure.
        baseline_samples: Optional baseline activations of shape ``[N, D]`` (``N`` inputs,
            ``D`` neurons). When provided, the visualization switches to a neuron-centric
            plot that scatters the base model activations for each neuron and overlays the
            defended activations.
        safety_samples: Optional activations from the safety defense aligned with
            ``baseline_samples`` (same ``N`` and ``D``).
        privacy_samples: Optional activations from the privacy defense aligned with
            ``baseline_samples`` (same ``N`` and ``D``).
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Wedge

    baseline_tensor = _coerce_sample_tensor(baseline_samples, "Baseline samples")
    safety_tensor = _coerce_sample_tensor(safety_samples, "Safety samples")
    privacy_tensor = _coerce_sample_tensor(privacy_samples, "Privacy samples")

    num_samples = None
    embedding_dim = None
    for name, tensor in (
        ("baseline", baseline_tensor),
        ("safety", safety_tensor),
        ("privacy", privacy_tensor),
    ):
        if tensor is None:
            continue
        if num_samples is None:
            num_samples = tensor.shape[0]
        elif tensor.shape[0] != num_samples:
            raise ValueError("All sample tensors must share the same number of rows (N).")
        if embedding_dim is None:
            embedding_dim = tensor.shape[1]
        elif tensor.shape[1] != embedding_dim:
            raise ValueError("All sample tensors must share the same embedding size (D).")

    sample_tensors = [t for t in (baseline_tensor, safety_tensor, privacy_tensor) if t is not None]
    if baseline_tensor is not None:
        plot_mode = "neurons"
    else:
        plot_mode = "concepts"

    try:
        style_ctx = plt.style.context("seaborn-v0_8-whitegrid")
    except OSError:
        style_ctx = nullcontext()

    with style_ctx:
        fig, ax = plt.subplots(figsize=(7.5, 7))

        if plot_mode == "concepts":
            basis_inputs = stack_concept_directions(safety, privacy)
            basis = compute_projection_basis(basis_inputs)

            def _project(tensor: torch.Tensor | None) -> torch.Tensor | None:
                return None if tensor is None else tensor @ basis.T

            directions = stack_concept_directions(safety, privacy)
            concept_coords = directions @ basis.T
            coords_list = concept_coords.tolist()
            labels = ["Safety concept", "Privacy concept"]
            colors = ["#2878b5", "#d14a61"]

            baseline_proj = _project(baseline_tensor)
            safety_proj = _project(safety_tensor)
            privacy_proj = _project(privacy_tensor)

            tensors_for_limit = [concept_coords]
            tensors_for_limit.extend(
                tensor
                for tensor in (baseline_proj, safety_proj, privacy_proj)
                if tensor is not None
            )
            limit = 1.0
            if tensors_for_limit:
                limit = max(
                    float(tensor.abs().max().item())
                    for tensor in tensors_for_limit
                    if tensor.numel()
                )
                limit = max(limit, 1e-3)
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
                ax.scatter(
                    [x],
                    [y],
                    color=color,
                    s=80,
                    zorder=5,
                    edgecolors="white",
                    linewidths=1.5,
                )
                ax.annotate(
                    label,
                    (x, y),
                    xytext=(6, 6),
                    textcoords="offset points",
                    color=color,
                    fontsize=12,
                    fontweight="bold",
                )

            legend_handles = []
            legend_labels = []

            if baseline_proj is not None:
                baseline_cloud = ax.scatter(
                    baseline_proj[:, 0],
                    baseline_proj[:, 1],
                    color="#888888",
                    s=18,
                    alpha=0.35,
                    label="Baseline samples",
                )
                legend_handles.append(baseline_cloud)
                legend_labels.append("Baseline cloud")

            per_arrow_handles = []
            per_arrow_labels = []

            def _draw_sample_arrows(
                start: torch.Tensor | None,
                end: torch.Tensor | None,
                color: str,
                linestyle: str = "-",
            ) -> bool:
                if start is None or end is None:
                    return False
                for idx in range(start.shape[0]):
                    arrow = FancyArrowPatch(
                        (float(start[idx, 0]), float(start[idx, 1])),
                        (float(end[idx, 0]), float(end[idx, 1])),
                        arrowstyle="->",
                        mutation_scale=8,
                        linewidth=1.0,
                        linestyle=linestyle,
                        color=color,
                        alpha=0.7,
                    )
                    ax.add_patch(arrow)
                return True

            if _draw_sample_arrows(baseline_proj, safety_proj, "#3c8dbc"):
                per_arrow_handles.append(
                    Line2D([0, 1], [0, 0], color="#3c8dbc", linestyle="-", linewidth=1.5)
                )
                per_arrow_labels.append("Safety sample shift")
            if _draw_sample_arrows(
                baseline_proj,
                privacy_proj,
                "#c9714d",
                linestyle="--",
            ):
                per_arrow_handles.append(
                    Line2D([0, 1], [0, 0], color="#c9714d", linestyle="--", linewidth=1.5)
                )
                per_arrow_labels.append("Privacy sample shift")

            def _plot_mean_arrow(
                displacement: torch.Tensor | None, color: str, label: str
            ):
                if displacement is None:
                    return None
                mean_vec = displacement.mean(dim=0)
                arrow = FancyArrowPatch(
                    (0.0, 0.0),
                    (float(mean_vec[0]), float(mean_vec[1])),
                    arrowstyle="-|>",
                    mutation_scale=25,
                    linewidth=4.0,
                    color=color,
                    alpha=0.9,
                )
                ax.add_patch(arrow)
                legend_handles.append(
                    Line2D([0, 1], [0, 0], color=color, linewidth=4.0, marker=">", markersize=8)
                )
                legend_labels.append(label)
                return mean_vec

            safety_disp = (
                safety_proj - baseline_proj
                if (safety_proj is not None and baseline_proj is not None)
                else None
            )
            privacy_disp = (
                privacy_proj - baseline_proj
                if (privacy_proj is not None and baseline_proj is not None)
                else None
            )

            mean_safety = _plot_mean_arrow(
                safety_disp, "#1c5d84", "Mean safety displacement"
            )
            mean_privacy = _plot_mean_arrow(
                privacy_disp, "#a24732", "Mean privacy displacement"
            )

            if mean_safety is not None and mean_privacy is not None:
                dot = float(torch.dot(mean_safety, mean_privacy))
                norm_prod = float(mean_safety.norm() * mean_privacy.norm())
                if norm_prod > 0:
                    cos_val = max(-1.0, min(1.0, dot / norm_prod))
                    mean_angle = math.degrees(math.acos(cos_val))
                    ax.text(
                        0.05 * limit,
                        0.05 * limit,
                        f"Δθ ≈ {mean_angle:.1f}°",
                        fontsize=11,
                        color="#333333",
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
                    )
                    start_angle = math.degrees(math.atan2(mean_safety[1], mean_safety[0]))
                    end_angle = math.degrees(math.atan2(mean_privacy[1], mean_privacy[0]))
                    wedge = Wedge(
                        (0.0, 0.0),
                        0.35 * limit,
                        min(start_angle, end_angle),
                        max(start_angle, end_angle),
                        width=0.05 * limit,
                        facecolor="#f6d9c6",
                        edgecolor="none",
                        alpha=0.6,
                    )
                    ax.add_patch(wedge)

            legend_handles.extend(per_arrow_handles)
            legend_labels.extend(per_arrow_labels)

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
            concept_handle = Line2D([0, 1], [0, 0], color="#2878b5", linewidth=3.0)
            privacy_handle = Line2D([0, 1], [0, 0], color="#d14a61", linewidth=3.0)
            legend_handles.extend([concept_handle, privacy_handle])
            legend_labels.extend(labels)
            ax.legend(legend_handles, legend_labels, loc="upper right", frameon=False)

        else:
            assert baseline_tensor is not None

            def _transpose(tensor: torch.Tensor | None) -> torch.Tensor | None:
                return None if tensor is None else tensor.T.contiguous()

            baseline_neurons = _transpose(baseline_tensor)
            safety_neurons = _transpose(safety_tensor)
            privacy_neurons = _transpose(privacy_tensor)

            neuron_mats = [m for m in (baseline_neurons, safety_neurons, privacy_neurons) if m is not None]
            if not neuron_mats:
                raise ValueError("Neuron activation plot requires at least baseline activations.")

            column_mean = torch.stack([m.mean(dim=0, keepdim=True) for m in neuron_mats]).mean(dim=0)

            def _center(mat: torch.Tensor | None) -> torch.Tensor | None:
                return None if mat is None else mat - column_mean

            centered_mats = [_center(mat) for mat in neuron_mats]
            basis_inputs = torch.cat(centered_mats, dim=0)
            basis = compute_projection_basis(basis_inputs)

            def _project_neurons(mat: torch.Tensor | None) -> torch.Tensor | None:
                return None if mat is None else (mat - column_mean) @ basis.T

            baseline_proj = _project_neurons(baseline_neurons)
            safety_proj = _project_neurons(safety_neurons)
            privacy_proj = _project_neurons(privacy_neurons)

            num_neurons = baseline_proj.shape[0]
            max_points = 600
            if num_neurons > max_points:
                idx = torch.linspace(0, num_neurons - 1, max_points)
                idx = torch.unique(idx.round().to(dtype=torch.long))
                if idx.numel() > max_points:
                    idx = idx[:max_points]
            else:
                idx = torch.arange(num_neurons)

            def _subset(mat: torch.Tensor | None) -> torch.Tensor | None:
                return None if mat is None else mat[idx]

            baseline_proj = _subset(baseline_proj)
            safety_proj = _subset(safety_proj)
            privacy_proj = _subset(privacy_proj)

            baseline_means = baseline_tensor.mean(dim=0)[idx]
            q80 = torch.quantile(baseline_means, 0.8).item()
            q50 = torch.quantile(baseline_means, 0.5).item()
            category_colors = {
                "strong": "#d73027",
                "moderate": "#fc8d59",
                "suppressed": "#1a9850",
            }

            def _category(value: float) -> str:
                if value >= q80:
                    return "strong"
                if value >= q50:
                    return "moderate"
                return "suppressed"

            neuron_categories = [_category(val.item()) for val in baseline_means]
            scatter_colors = [category_colors[cat] for cat in neuron_categories]

            legend_handles: list[Line2D] = []
            legend_labels: list[str] = []

            baseline_scatter = ax.scatter(
                baseline_proj[:, 0],
                baseline_proj[:, 1],
                c=scatter_colors,
                s=26,
                alpha=0.85,
                marker="o",
                edgecolors="#fdfdfd",
                linewidths=0.6,
            )
            # Add categorical legend entries
            for cat, color in category_colors.items():
                if cat not in neuron_categories:
                    continue
                legend_handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markeredgecolor="#fdfdfd", markersize=7))
                if cat == "strong":
                    label = "强激活 (Top 20%)"
                elif cat == "moderate":
                    label = "中等激活"
                else:
                    label = "被抑制/低激活"
                legend_labels.append(label)

            def _plot_defense(
                defense_proj: torch.Tensor | None,
                color: str,
                marker: str,
                label: str,
                linestyle: str,
            ) -> torch.Tensor | None:
                if defense_proj is None:
                    return None
                ax.scatter(
                    defense_proj[:, 0],
                    defense_proj[:, 1],
                    color=color,
                    s=35,
                    marker=marker,
                    alpha=0.9,
                    edgecolors="#fdfdfd",
                    linewidths=0.6,
                )
                for i in range(defense_proj.shape[0]):
                    ax.plot(
                        [float(baseline_proj[i, 0]), float(defense_proj[i, 0])],
                        [float(baseline_proj[i, 1]), float(defense_proj[i, 1])],
                        linestyle=linestyle,
                        linewidth=0.9,
                        color=color,
                        alpha=0.6,
                    )
                legend_handles.append(
                    Line2D([0, 1], [0, 0], color=color, linestyle=linestyle, linewidth=1.2)
                )
                legend_labels.append(f"{label} 位移")
                return defense_proj - baseline_proj

            safety_disp = _plot_defense(
                safety_proj,
                color="#2b83ba",
                marker="^",
                label="Safety 防御",
                linestyle="-.",
            )
            privacy_disp = _plot_defense(
                privacy_proj,
                color="#fdae61",
                marker="s",
                label="Privacy 防御",
                linestyle=":",
            )

            def _plot_mean_shift(disp: torch.Tensor | None, color: str, label: str):
                if disp is None:
                    return None
                mean_vec = disp.mean(dim=0)
                arrow = FancyArrowPatch(
                    (0.0, 0.0),
                    (float(mean_vec[0]), float(mean_vec[1])),
                    arrowstyle="-|>",
                    mutation_scale=22,
                    linewidth=3.2,
                    color=color,
                    alpha=0.85,
                )
                ax.add_patch(arrow)
                legend_handles.append(
                    Line2D([0, 1], [0, 0], color=color, linewidth=3.2, marker=">", markersize=7)
                )
                legend_labels.append(label)
                return mean_vec

            mean_safety = _plot_mean_shift(safety_disp, "#1a237e", "Safety 均值位移")
            mean_privacy = _plot_mean_shift(privacy_disp, "#c43c00", "Privacy 均值位移")

            if mean_safety is not None and mean_privacy is not None:
                dot = float(torch.dot(mean_safety, mean_privacy))
                norm_prod = float(mean_safety.norm() * mean_privacy.norm())
                if norm_prod > 0:
                    cos_val = max(-1.0, min(1.0, dot / norm_prod))
                    angle = math.degrees(math.acos(cos_val))
                    ax.annotate(
                        f"夹角 ≈ {angle:.1f}°",
                        xy=(0.02, 0.05),
                        xycoords="axes fraction",
                        fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
                    )

            tensors_for_limit = [baseline_proj]
            tensors_for_limit.extend(
                tensor for tensor in (safety_proj, privacy_proj) if tensor is not None
            )
            limit = max(
                float(tensor.abs().max().item())
                for tensor in tensors_for_limit
                if tensor.numel()
            )
            limit = max(limit * 1.2, 1e-3)

            ax.set_xlabel("激活分量 1", fontsize=12)
            ax.set_ylabel("激活分量 2", fontsize=12)
            ax.set_title("每个神经元的激活变化", fontsize=14, fontweight="bold")
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal", "box")
            ax.axhline(0, color="#aaaaaa", linewidth=1.0, alpha=0.5)
            ax.axvline(0, color="#aaaaaa", linewidth=1.0, alpha=0.5)
            ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
            ax.legend(legend_handles, legend_labels, loc="upper right", frameon=False)

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
