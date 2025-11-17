"""Scatter plot utilities for safety/privacy activation projections."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ConflictProjectionData:
    """Container storing 2D projection coordinates for all activation groups."""

    base_safety: np.ndarray
    defense_safety: np.ndarray
    base_privacy: np.ndarray
    defense_privacy: np.ndarray


def _ensure_path(path: Path | str) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _centroid(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        raise ValueError("Cannot compute centroids for empty projection arrays")
    return points.mean(axis=0)


def plot_conflict_projection(
    projections: ConflictProjectionData,
    output_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Render a scatter plot contrasting base/defense activations."""

    output = _ensure_path(output_path)
    fig, ax = plt.subplots(figsize=(8, 6))

    groups = [
        (projections.base_safety, "Base – Safety", "tab:blue", "o"),
        (projections.defense_safety, "Defense – Safety", "tab:orange", "s"),
        (projections.base_privacy, "Base – Privacy", "tab:green", "^"),
        (projections.defense_privacy, "Defense – Privacy", "tab:red", "D"),
    ]

    for coords, label, color, marker in groups:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            label=label,
            c=color,
            marker=marker,
            alpha=0.75,
            edgecolors="white",
            linewidths=0.4,
        )

    # Draw centroid arrows showing the shift introduced by each defense.
    base_safety_center = _centroid(projections.base_safety)
    defense_safety_center = _centroid(projections.defense_safety)
    base_privacy_center = _centroid(projections.base_privacy)
    defense_privacy_center = _centroid(projections.defense_privacy)

    def _draw_shift(start: np.ndarray, end: np.ndarray, color: str, label: str) -> None:
        delta = end - start
        ax.arrow(
            start[0],
            start[1],
            delta[0],
            delta[1],
            color=color,
            width=0.002,
            head_width=0.12,
            length_includes_head=True,
            alpha=0.9,
            label=label,
        )
        midpoint = start + delta * 0.6
        ax.annotate(
            label,
            xy=(midpoint[0], midpoint[1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize="medium",
            color=color,
            weight="bold",
        )

    _draw_shift(base_safety_center, defense_safety_center, "black", "Safety shift")
    _draw_shift(base_privacy_center, defense_privacy_center, "tab:purple", "Privacy shift")

    ax.set_xlabel("Projection dimension 1")
    ax.set_ylabel("Projection dimension 2")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output
