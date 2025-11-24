"""Concept vector construction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Tuple

import torch
from sklearn.decomposition import PCA


@dataclass
class ConceptVector:
    """Stores a normalized concept vector and metadata."""

    direction: torch.Tensor
    method: str
    explained_variance: float | None = None
    raw_norm: float | None = None

    def project(self, activations: torch.Tensor) -> torch.Tensor:
        """Project activations onto the concept direction."""

        return torch.matmul(activations, self.direction)


def aggregate_difference(
    delta_activations: torch.Tensor,
    method: Literal["mean", "pca"] = "mean",
) -> ConceptVector:
    """Aggregate activation differences into a single concept vector."""

    flat = delta_activations.view(delta_activations.size(0), -1)
    if method == "mean":
        direction = flat.mean(dim=0)
        norm = direction.norm(p=2)
        direction = direction / (norm + 1e-12)
        return ConceptVector(
            direction=direction,
            method="mean",
            explained_variance=None,
            raw_norm=float(norm),
        )
    if method == "pca":
        pca = PCA(n_components=1)
        pca.fit(flat.cpu().numpy())
        direction = torch.from_numpy(pca.components_[0]).to(delta_activations.device)
        norm = direction.norm(p=2)
        direction = direction / (norm + 1e-12)
        return ConceptVector(
            direction=direction,
            method="pca",
            explained_variance=float(pca.explained_variance_ratio_[0]),
            raw_norm=float(norm),
        )
    raise ValueError(f"Unknown aggregation method: {method}")


def purify_concept_vector(
    mixed: ConceptVector,
    nuisance: ConceptVector,
) -> ConceptVector:
    """Remove nuisance components from a mixed concept vector."""

    projection = torch.dot(mixed.direction, nuisance.direction)
    purified_direction = mixed.direction - projection * nuisance.direction
    norm = purified_direction.norm(p=2)
    purified_direction = purified_direction / (norm + 1e-12)
    return ConceptVector(
        direction=purified_direction,
        method="purified",
        explained_variance=None,
        raw_norm=float(norm),
    )


def cosine_similarity(vec_a: ConceptVector, vec_b: ConceptVector) -> float:
    """Compute cosine similarity between two concept vectors."""

    return torch.dot(vec_a.direction, vec_b.direction).item()
