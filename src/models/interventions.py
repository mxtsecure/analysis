"""Causal intervention utilities for concept vectors."""
from __future__ import annotations

from contextlib import contextmanager
from typing import List

import torch
from torch import nn

from .activations import get_module_by_name
from ..analysis.concept_vectors import ConceptVector


@contextmanager
def linear_intervention(
    model: nn.Module,
    module_name: str,
    concept: ConceptVector,
    scale: float,
):
    """Context manager that injects a linear intervention at ``module_name``."""

    module = get_module_by_name(model, module_name)
    direction = concept.direction.to(next(model.parameters()).device)

    def hook(_module, inputs, output):  # pragma: no cover - hook
        shape = output.shape
        flat = output.view(shape[0], -1)
        flat = flat + scale * direction
        return flat.view(shape)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def evaluate_intervention(
    model: nn.Module,
    dataloader,
    module_name: str,
    concept: ConceptVector,
    scale: float,
    metric_fn,
    device: torch.device,
) -> float:
    """Apply a linear intervention and return the aggregated metric value."""

    scores: List[float] = []
    with linear_intervention(model, module_name, concept, scale):
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "metadata"}
            outputs = model.generate(**inputs)
            scores.append(metric_fn(outputs, batch))
    return float(sum(scores) / len(scores))
