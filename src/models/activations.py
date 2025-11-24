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


def _select_final_token(
    hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    """Return the activation at the final token position for each sample.

    If an ``attention_mask`` is available, the last non-padding token is used per
    sequence. Otherwise, the activations from the final sequence position are
    selected.
    """

    if hidden_states.dim() < 2:
        raise ValueError("Hidden states must have at least batch and sequence dimensions")

    if hidden_states.dim() == 2:
        # Already (batch, hidden_dim); nothing to slice.
        return hidden_states

    if attention_mask is not None:
        if attention_mask.dim() != 2:
            raise ValueError("Attention mask must have shape (batch, seq_len)")
        if attention_mask.size(0) != hidden_states.size(0):
            raise ValueError("Attention mask batch size must match hidden states")

        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0).unsqueeze(-1).unsqueeze(-1)
        lengths = lengths.expand(-1, 1, hidden_states.size(-1))
        return hidden_states.gather(1, lengths).squeeze(1)

    return hidden_states[:, -1, ...]


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

    if base.attention_mask is not None and finetuned.attention_mask is not None:
        if base.attention_mask.shape != finetuned.attention_mask.shape:
            raise ValueError("Attention masks must have matching shapes for differencing")
        attention_mask = base.attention_mask
    else:
        attention_mask = base.attention_mask or finetuned.attention_mask

    base_final = _select_final_token(base.hidden_states, attention_mask)
    finetuned_final = _select_final_token(finetuned.hidden_states, attention_mask)
    return finetuned_final - base_final
