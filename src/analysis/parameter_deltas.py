from __future__ import annotations

"""Granular parameter delta analysis across transformer checkpoints."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from transformers import AutoModelForCausalLM


@dataclass
class ParameterDeltaSummary:
    """Structured collection of per-parameter and aggregated delta statistics."""

    parameters: pd.DataFrame
    layers: pd.DataFrame
    modules: pd.DataFrame
    submodules: pd.DataFrame


def _prepare_tensor(tensor: Tensor) -> Tensor:
    if tensor.device.type != "cpu":
        tensor = tensor.detach().cpu()
    else:
        tensor = tensor.detach()
    if tensor.dtype not in (torch.float32, torch.float64):
        tensor = tensor.float()
    return tensor


def _iterate_common_parameters(
    base_state: Mapping[str, Tensor],
    finetuned_state: Mapping[str, Tensor],
) -> Iterable[tuple[str, Tensor, Tensor]]:
    common = sorted(set(base_state.keys()) & set(finetuned_state.keys()))
    for name in common:
        yield name, _prepare_tensor(base_state[name]), _prepare_tensor(finetuned_state[name])


def _extract_layer_module(name: str) -> tuple[Optional[str], Optional[str], Optional[str], Optional[int]]:
    parts = name.split(".")
    for idx in range(len(parts) - 1):
        if parts[idx] == "layers" and idx > 0 and parts[idx - 1] == "model":
            if idx + 1 >= len(parts):
                break
            layer_id = parts[idx + 1]
            if not layer_id.isdigit():
                break
            layer = f"model.layers.{layer_id}"
            layer_index = int(layer_id)
            module = None
            submodule = None
            if idx + 2 < len(parts):
                module = f"{layer}.{parts[idx + 2]}"
            if idx + 3 < len(parts):
                submodule = f"{module}.{parts[idx + 3]}" if module is not None else None
            return layer, module, submodule, layer_index
    return None, None, None, None


def compute_parameter_deltas(
    base_state: Mapping[str, Tensor],
    finetuned_state: Mapping[str, Tensor],
    *,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """Compute per-parameter delta statistics between two checkpoints."""

    records = []
    for name, base_param, finetuned_param in _iterate_common_parameters(base_state, finetuned_state):
        delta = finetuned_param - base_param
        abs_delta = delta.abs()
        mean_abs = float(abs_delta.mean().item())
        mean_l2 = float(torch.sqrt(torch.mean(delta.pow(2))).item())
        denom = base_param.abs() + eps
        mean_abs_rel = float((abs_delta / denom).mean().item())
        numel = delta.numel()
        layer, module, submodule, layer_index = _extract_layer_module(name)
        records.append(
            {
                "parameter": name,
                "numel": numel,
                "mean_abs": mean_abs,
                "mean_l2": mean_l2,
                "mean_abs_rel": mean_abs_rel,
                "layer": layer,
                "layer_index": layer_index,
                "module": module,
                "submodule": submodule,
            }
        )
    df = pd.DataFrame.from_records(records)
    return df


def _weighted_average(values: Sequence[float], weights: Sequence[int]) -> float:
    if not weights:
        return 0.0
    arr_values = np.asarray(values, dtype=np.float64)
    arr_weights = np.asarray(weights, dtype=np.float64)
    total = arr_weights.sum()
    if total == 0:
        return float(arr_values.mean() if arr_values.size else 0.0)
    return float(np.average(arr_values, weights=arr_weights))


def _aggregate(df: pd.DataFrame, key: str) -> pd.DataFrame:
    if key not in df.columns:
        raise KeyError(f"DataFrame does not contain '{key}' column")

    groups = []
    for value, group in df.groupby(key, dropna=True):
        weights = group["numel"].tolist()
        record = {
            key: value,
            "numel": int(sum(weights)),
            "mean_abs": _weighted_average(group["mean_abs"].tolist(), weights),
            "mean_l2": _weighted_average(group["mean_l2"].tolist(), weights),
            "mean_abs_rel": _weighted_average(group["mean_abs_rel"].tolist(), weights),
        }
        if "layer_index" in group.columns and not pd.isna(group["layer_index"].iloc[0]):
            # keep layer index for layer-level aggregations
            if key == "layer":
                record["layer_index"] = int(group["layer_index"].iloc[0])
            elif key in {"module", "submodule"}:
                indices = group["layer_index"].dropna().unique()
                record["layer_index"] = int(indices[0]) if len(indices) == 1 else np.nan
        groups.append(record)
    result = pd.DataFrame(groups)
    if "layer_index" in result.columns:
        result = result.sort_values([col for col in ["layer_index", key] if col in result.columns])
    else:
        result = result.sort_values(key)
    return result.reset_index(drop=True)


def summarise_parameter_deltas(
    base_state: Mapping[str, Tensor],
    finetuned_state: Mapping[str, Tensor],
) -> ParameterDeltaSummary:
    """Compute per-parameter, per-layer, and per-module delta statistics."""

    parameters = compute_parameter_deltas(base_state, finetuned_state)
    layers = _aggregate(parameters.dropna(subset=["layer"]), "layer") if not parameters.empty else pd.DataFrame()
    modules = _aggregate(parameters.dropna(subset=["module"]), "module") if not parameters.empty else pd.DataFrame()
    submodules = _aggregate(parameters.dropna(subset=["submodule"]), "submodule") if not parameters.empty else pd.DataFrame()
    return ParameterDeltaSummary(
        parameters=parameters,
        layers=layers,
        modules=modules,
        submodules=submodules,
    )


def load_state_dict(
    model_path: str | Path,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device: str | torch.device = "cpu",
) -> Dict[str, Tensor]:
    """Load a causal LM checkpoint and return a CPU state dictionary."""

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    model.to(device)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model.cpu()
    del model
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return state


def analyze_checkpoints(
    base_model_path: str | Path,
    finetuned_model_path: str | Path,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device: str | torch.device = "cpu",
) -> ParameterDeltaSummary:
    """Load two checkpoints and compute granular parameter delta statistics."""

    base_state = load_state_dict(base_model_path, torch_dtype=torch_dtype, device=device)
    finetuned_state = load_state_dict(finetuned_model_path, torch_dtype=torch_dtype, device=device)
    return summarise_parameter_deltas(base_state, finetuned_state)
