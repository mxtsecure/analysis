"""Activation extraction utilities.

These helpers implement the measurement pipeline required by Step 1 of the research plan.
They support forwarding datasets through a base model and a fine-tuned variant while
capturing intermediate layer activations that can later be aggregated into concept
vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.hooks import RemovableHandle
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


def forward_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    module_name: Union[str, Sequence[str]],
    device: torch.device,
    description: str,
    *,
    max_batches: int | None = None,
) -> ActivationBatch | Dict[str, ActivationBatch]:
    """Forward the dataset through ``model`` and collect activations."""

    if isinstance(module_name, str):
        module_names = [module_name]
    else:
        module_names = list(module_name)
    if not module_names:
        raise ValueError("At least one module name must be provided for activation capture")

    modules = {name: get_module_by_name(model, name) for name in module_names}
    collected: Dict[str, List[torch.Tensor]] = {name: [] for name in module_names}
    attention_masks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        handles: List[RemovableHandle] = []
        try:
            for name, module in modules.items():
                storage = collected[name]

                def hook(_module, inputs, output, *, _storage=storage):
                    if isinstance(output, tuple):
                        tensor = output[0]
                    elif isinstance(output, dict):
                        tensor = output["hidden_states"]
                    else:
                        tensor = output
                    _storage.append(tensor.detach())

                handles.append(module.register_forward_hook(hook))

            for batch_idx, batch in enumerate(tqdm(dataloader, desc=description)):
                inputs = {k: v.to(device) for k, v in batch.items() if k != "metadata"}
                model(**inputs)
                if "attention_mask" in inputs:
                    attention_masks.append(inputs["attention_mask"].detach().cpu())
                if max_batches is not None and batch_idx + 1 >= max_batches:
                    break
        finally:
            for handle in handles:
                handle.remove()

    hidden_states = {
        name: torch.cat([tensor.cpu() for tensor in storage], dim=0)
        for name, storage in collected.items()
    }
    attention_mask = (
        torch.cat(attention_masks, dim=0) if attention_masks else None
    )
    batches = {
        name: ActivationBatch(hidden_states=tensor, attention_mask=attention_mask)
        for name, tensor in hidden_states.items()
    }
    if len(module_names) == 1:
        return batches[module_names[0]]
    return batches


def compute_activation_difference(
    base: ActivationBatch,
    finetuned: ActivationBatch,
) -> torch.Tensor:
    """Compute the element-wise difference between two activation batches."""

    if base.hidden_states.shape != finetuned.hidden_states.shape:
        raise ValueError("Activation tensors must have the same shape for differencing")
    return finetuned.hidden_states - base.hidden_states
