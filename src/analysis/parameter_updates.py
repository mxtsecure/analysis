"""Utilities for computing parameter update vectors."""
from __future__ import annotations
from typing import Dict, Iterable, Tuple

import torch
from torch import nn


def state_dict_to_vector(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten a state dictionary into a single 1-D tensor."""

    vectors = []
    for tensor in state_dict.values():
        vectors.append(tensor.reshape(-1))
    return torch.cat(vectors)


def parameter_update_vector(
    base_model: nn.Module,
    finetuned_model: nn.Module,
) -> torch.Tensor:
    """Compute the parameter update vector Δθ."""

    base_state = base_model.state_dict()
    finetuned_state = finetuned_model.state_dict()
    deltas = []
    for key in base_state.keys():
        deltas.append((finetuned_state[key] - base_state[key]).reshape(-1))
    return torch.cat(deltas)


def parameter_update_vector_from_state(
    base_state: Dict[str, torch.Tensor],
    finetuned_state: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute the parameter update vector Δθ from two state dictionaries."""

    deltas = []
    for key in base_state.keys():
        if key not in finetuned_state:
            continue
        deltas.append((finetuned_state[key] - base_state[key]).reshape(-1))
    return torch.cat(deltas) if deltas else torch.tensor([])


def cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute cosine similarity between two parameter vectors."""

    return torch.dot(vec_a, vec_b).item() / (
        vec_a.norm(p=2) * vec_b.norm(p=2) + 1e-12
    )


def projection_onto(vec_target: torch.Tensor, vec_reference: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Scalar projection of ``vec_target`` onto ``vec_reference``.

    A positive value indicates that ``vec_target`` moves in roughly the same direction as
    ``vec_reference``; a negative value indicates opposition. The result has the same
    units as the parameter updates (i.e., it is not normalised by the target norm).
    """

    ref_norm = vec_reference.norm(p=2)
    return torch.dot(vec_target, vec_reference).item() / (ref_norm.item() + eps)
