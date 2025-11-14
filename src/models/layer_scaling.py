"""Utilities for applying causal interventions via layer scaling."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass
class LayerRange:
    """Description of the layers that should be scaled."""

    start: int
    end: Optional[int] = None

    def indices(self, total_layers: int) -> range:
        """Return the layer indices covered by this range."""

        stop = total_layers if self.end is None else min(self.end, total_layers)
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if stop <= self.start:
            raise ValueError("end must be greater than start")
        return range(self.start, stop)


def _resolve_layers(model) -> List[torch.nn.Module]:
    """Extract the sequential decoder layers from ``model``."""

    candidates: Sequence[str] = (
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
        "gpt_neox.layers",
    )
    for candidate in candidates:
        current = model
        parts = candidate.split(".")
        for part in parts:
            if not hasattr(current, part):
                break
            current = getattr(current, part)
        else:
            layers = list(current)
            if not layers:
                continue
            return layers
    raise AttributeError("Unable to locate transformer layers on the provided model")


def scale_model_layers(
    base_model,
    layer_range: LayerRange,
    scale_factor: float,
    reference_model=None,
) -> torch.nn.Module:
    """Create a scaled copy of ``base_model``."""

    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")

    cloned = copy.deepcopy(base_model)
    source = base_model if reference_model is None else reference_model

    target_layers = _resolve_layers(cloned)
    source_layers = _resolve_layers(source)
    indices = layer_range.indices(len(target_layers))

    with torch.no_grad():
        for idx in indices:
            target_layer = target_layers[idx]
            source_layer = source_layers[idx]
            source_params = dict(source_layer.named_parameters())
            for name, param in target_layer.named_parameters():
                ref_param = source_params.get(name)
                if ref_param is None:
                    continue
                param.copy_(ref_param * scale_factor)
    return cloned
