"""Concept gradient computation utilities."""
from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..models.activations import capture_activations, get_module_by_name
from .concept_vectors import ConceptVector


def flatten_gradients(parameters: List[nn.Parameter]) -> torch.Tensor:
    """Flatten gradients from model parameters into a single vector."""

    grads = []
    for param in parameters:
        if param.grad is None:
            grads.append(torch.zeros_like(param).reshape(-1))
        else:
            grads.append(param.grad.reshape(-1))
    return torch.cat(grads)


def compute_concept_gradient(
    model: nn.Module,
    dataloader: DataLoader,
    module_name: str,
    concept: ConceptVector,
    device: torch.device,
    desc: str,
) -> torch.Tensor:
    """Compute gradients induced by the proxy loss for ``concept``."""

    parameters = list(model.parameters())
    vector = concept.direction.to(device)
    module = get_module_by_name(model, module_name)
    gradients: List[torch.Tensor] = []

    model.train()
    for batch in tqdm(dataloader, desc=desc):
        model.zero_grad()
        activations: List[torch.Tensor] = []
        with capture_activations(module, activations):
            outputs = model(**{k: v.to(device) for k, v in batch.items() if k != "metadata"})
        activation = activations[0]
        flat = activation.view(activation.size(0), -1)
        loss = -(flat @ vector).mean()
        loss.backward()
        gradients.append(flatten_gradients(parameters).detach().cpu())
    return torch.stack(gradients).mean(dim=0)


def cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """Compute cosine similarity between two flattened gradient vectors."""

    return torch.dot(vec_a, vec_b).item() / (
        vec_a.norm(p=2) * vec_b.norm(p=2) + 1e-12
    )
