"""Activation extraction utilities.

These helpers implement the measurement pipeline required by Step 1 of the research plan.
They support forwarding datasets through a base model and a fine-tuned variant while
capturing intermediate layer activations that can later be aggregated into concept
vectors.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class ActivationBatch:
    """Container storing activation tensors for a batch of samples."""

    hidden_states: torch.Tensor
    attention_mask: Optional[torch.Tensor] = None


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    """Resolve a ``module_name`` string into the corresponding sub-module."""

    module = model
    for attr in module_name.split("."):
        module = getattr(module, attr)
    return module


@contextmanager
def capture_activations(module: nn.Module, storage: List[torch.Tensor]):
    """Register a forward hook that stores activations in ``storage``."""

    def hook(_module, inputs, output):  # pragma: no cover - simple hook
        if isinstance(output, tuple):
            tensor = output[0]
        elif isinstance(output, dict):
            tensor = output["hidden_states"]
        else:
            tensor = output
        storage.append(tensor.detach())

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def forward_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    module_name: str,
    device: torch.device,
    description: str,
) -> ActivationBatch:
    """Forward the dataset through ``model`` and collect activations."""

    module = get_module_by_name(model, module_name)
    collected: List[torch.Tensor] = []
    attention_masks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        with capture_activations(module, collected):
            for batch in tqdm(dataloader, desc=description):
                inputs = {k: v.to(device) for k, v in batch.items() if k != "metadata"}
                outputs = model(**inputs)
                if "attention_mask" in inputs:
                    attention_masks.append(inputs["attention_mask"].detach().cpu())
    hidden_states = torch.cat([tensor.cpu() for tensor in collected], dim=0)
    attention_mask = (
        torch.cat(attention_masks, dim=0) if attention_masks else None
    )
    return ActivationBatch(hidden_states=hidden_states, attention_mask=attention_mask)


def compute_activation_difference(
    base: ActivationBatch,
    finetuned: ActivationBatch,
) -> torch.Tensor:
    """Compute the element-wise difference between two activation batches."""

    if base.hidden_states.shape != finetuned.hidden_states.shape:
        raise ValueError("Activation tensors must have the same shape for differencing")
    return finetuned.hidden_states - base.hidden_states
