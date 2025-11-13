"""Utilities for identifying key representation layers for safety/privacy defenses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class CosineCurves:
    """Container storing cosine-based statistics for each transformer layer."""

    mu_nn: List[float]
    mu_nr: List[float]
    std_nn: List[float]
    std_nr: List[float]
    angle_nn: List[float]
    angle_nr: List[float]
    delta_phi: List[float]
    smooth_delta_phi: List[float]
    k_start: int
    k_peak: int
    k_end_rep: int


@dataclass
class ParameterCurves:
    """Stores mean absolute parameter deltas for attention/MLP blocks by layer."""

    s_attn: List[float]
    s_mlp: List[float]
    s_total: List[float]
    s_zscore: List[float]


@dataclass
class KeyLayerIntervals:
    """Represents the candidate and filtered key-layer intervals."""

    representational: Tuple[int, int]
    key: Optional[Tuple[int, int]]


@dataclass
class KeyLayerAnalysisResult:
    """Aggregates all statistics needed to report key layer regions."""

    cosine: CosineCurves
    parameters: ParameterCurves
    intervals: KeyLayerIntervals

    def to_dict(self) -> Dict[str, object]:
        """Convert the analysis result into a JSON-serialisable dictionary."""

        return {
            "cosine": {
                "mu_nn": self.cosine.mu_nn,
                "mu_nr": self.cosine.mu_nr,
                "std_nn": self.cosine.std_nn,
                "std_nr": self.cosine.std_nr,
                "angle_nn": self.cosine.angle_nn,
                "angle_nr": self.cosine.angle_nr,
                "delta_phi": self.cosine.delta_phi,
                "smooth_delta_phi": self.cosine.smooth_delta_phi,
                "k_start": self.cosine.k_start,
                "k_peak": self.cosine.k_peak,
                "k_end_rep": self.cosine.k_end_rep,
            },
            "parameters": {
                "s_attn": self.parameters.s_attn,
                "s_mlp": self.parameters.s_mlp,
                "s_total": self.parameters.s_total,
                "s_zscore": self.parameters.s_zscore,
            },
            "intervals": {
                "representational": list(self.intervals.representational),
                "key": list(self.intervals.key) if self.intervals.key is not None else None,
            },
        }


def _prepare_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    tensor_batch: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            tensor_batch[key] = value.to(device)
    return tensor_batch


def collect_last_token_hidden_states(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> List[torch.Tensor]:
    """Collect final token hidden states for every transformer block."""

    model.eval()
    hidden_accumulator: List[List[torch.Tensor]] = []
    with torch.no_grad():
        for batch in dataloader:
            tensor_batch = _prepare_batch(batch, device)
            attention_mask = tensor_batch["attention_mask"].long()
            outputs = model(
                **tensor_batch,
                output_hidden_states=True,
                use_cache=False,
            )
            # Hugging Face returns embeddings in hidden_states[0]; skip it.
            hidden_states = outputs.hidden_states[1:]
            if not hidden_accumulator:
                hidden_accumulator = [[] for _ in hidden_states]
            lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(attention_mask.size(0), device=device)
            for layer_idx, layer_hidden in enumerate(hidden_states):
                final_token = layer_hidden[batch_indices, lengths]
                hidden_accumulator[layer_idx].append(final_token.cpu())

    return [torch.cat(chunks, dim=0) for chunks in hidden_accumulator]


def _sample_pair_indices(
    size: int,
    num_pairs: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    first = torch.randint(size, (num_pairs,), generator=generator)
    second = torch.randint(size - 1, (num_pairs,), generator=generator)
    second = second + (second >= first)
    return first, second


def compute_cosine_curves(
    normal_states: Sequence[torch.Tensor],
    risk_states: Sequence[torch.Tensor],
    *,
    num_pairs: int = 500,
    seed: Optional[int] = None,
    smoothing: int = 5,
    baseline_layers: int = 3,
) -> CosineCurves:
    """Compute cosine/angle statistics for N–N and N–R comparisons."""

    if not normal_states:
        raise ValueError("normal_states must contain at least one layer")
    if len(normal_states) != len(risk_states):
        raise ValueError("normal_states and risk_states must have the same length")

    num_layers = len(normal_states)
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    nn_first, nn_second = _sample_pair_indices(normal_states[0].size(0), num_pairs, gen)
    nr_normal = torch.randint(normal_states[0].size(0), (num_pairs,), generator=gen)
    nr_risk = torch.randint(risk_states[0].size(0), (num_pairs,), generator=gen)

    mu_nn = []
    mu_nr = []
    std_nn = []
    std_nr = []
    angle_nn = []
    angle_nr = []

    cos_delta = []

    for layer_idx in range(num_layers):
        normal_layer = normal_states[layer_idx]
        risk_layer = risk_states[layer_idx]

        vec_nn_a = normal_layer[nn_first]
        vec_nn_b = normal_layer[nn_second]
        vec_nr_a = normal_layer[nr_normal]
        vec_nr_b = risk_layer[nr_risk]

        cos_nn = F.cosine_similarity(vec_nn_a, vec_nn_b, dim=1)
        cos_nr = F.cosine_similarity(vec_nr_a, vec_nr_b, dim=1)

        mu_nn.append(cos_nn.mean().item())
        mu_nr.append(cos_nr.mean().item())
        std_nn.append(cos_nn.std(unbiased=False).item())
        std_nr.append(cos_nr.std(unbiased=False).item())

        clamped_nn = torch.clamp(cos_nn, -1.0, 1.0)
        clamped_nr = torch.clamp(cos_nr, -1.0, 1.0)
        ang_nn = torch.rad2deg(torch.acos(clamped_nn)).mean().item()
        ang_nr = torch.rad2deg(torch.acos(clamped_nr)).mean().item()
        angle_nn.append(ang_nn)
        angle_nr.append(ang_nr)
        cos_delta.append(ang_nr - ang_nn)

    delta_phi = np.array(cos_delta)
    window = max(1, smoothing)
    kernel = np.ones(window, dtype=np.float64) / window
    smooth_delta = np.convolve(delta_phi, kernel, mode="same")

    baseline_end = min(baseline_layers, len(smooth_delta))
    baseline_values = smooth_delta[:baseline_end]
    mu0 = float(np.mean(baseline_values))
    std0 = float(np.std(baseline_values) + 1e-12)

    start_threshold = mu0 + 2.0 * std0
    candidate_indices = np.where(smooth_delta > start_threshold)[0]
    if candidate_indices.size == 0:
        k_start = 0
    else:
        k_start = int(candidate_indices[0])

    if k_start >= len(smooth_delta):
        k_start = len(smooth_delta) - 1

    k_peak = int(k_start + np.argmax(smooth_delta[k_start:]))

    end_threshold = mu0 + 1.5 * std0
    half_peak = 0.5 * smooth_delta[k_peak]
    k_end_rep = len(smooth_delta) - 1
    for idx in range(k_peak, len(smooth_delta)):
        if smooth_delta[idx] < end_threshold or smooth_delta[idx] < half_peak:
            k_end_rep = idx
            break

    return CosineCurves(
        mu_nn=mu_nn,
        mu_nr=mu_nr,
        std_nn=std_nn,
        std_nr=std_nr,
        angle_nn=angle_nn,
        angle_nr=angle_nr,
        delta_phi=cos_delta,
        smooth_delta_phi=smooth_delta.tolist(),
        k_start=k_start,
        k_peak=k_peak,
        k_end_rep=k_end_rep,
    )


def _initialise_stats(num_layers: int) -> Tuple[List[float], List[int]]:
    return [0.0 for _ in range(num_layers)], [0 for _ in range(num_layers)]


def compute_parameter_curves(
    base_state: Dict[str, torch.Tensor],
    defense_state: Dict[str, torch.Tensor],
    num_layers: int,
    z_threshold: float = 1.0,
) -> Tuple[ParameterCurves, List[int]]:
    """Compute mean absolute deltas for attention and MLP blocks."""

    attn_sum, attn_count = _initialise_stats(num_layers)
    mlp_sum, mlp_count = _initialise_stats(num_layers)

    for name, base_param in base_state.items():
        if name not in defense_state:
            continue
        if not name.startswith("model.layers."):
            continue
        remainder = name[len("model.layers.") :]
        layer_str, _, sub_name = remainder.partition(".")
        if not layer_str.isdigit():
            continue
        layer_idx = int(layer_str)
        if layer_idx >= num_layers:
            continue

        delta = defense_state[name] - base_param
        target_sum: Optional[List[float]] = None
        target_count: Optional[List[int]] = None
        if ".self_attn." in sub_name or sub_name.startswith("self_attn"):
            target_sum, target_count = attn_sum, attn_count
        elif ".mlp." in sub_name or sub_name.startswith("mlp"):
            target_sum, target_count = mlp_sum, mlp_count
        else:
            continue

        target_sum[layer_idx] += delta.abs().sum().item()
        target_count[layer_idx] += delta.numel()

    s_attn = [
        (attn_sum[idx] / attn_count[idx]) if attn_count[idx] else 0.0
        for idx in range(num_layers)
    ]
    s_mlp = [
        (mlp_sum[idx] / mlp_count[idx]) if mlp_count[idx] else 0.0
        for idx in range(num_layers)
    ]
    s_total = [a + b for a, b in zip(s_attn, s_mlp)]

    mean_total = float(np.mean(s_total))
    std_total = float(np.std(s_total) + 1e-12)
    s_z = [float((value - mean_total) / std_total) for value in s_total]

    significant_layers = [idx for idx, score in enumerate(s_z) if score >= z_threshold]

    return (
        ParameterCurves(s_attn=s_attn, s_mlp=s_mlp, s_total=s_total, s_zscore=s_z),
        significant_layers,
    )


def _longest_consecutive_segment(indices: Sequence[int]) -> Optional[Tuple[int, int]]:
    if not indices:
        return None
    sorted_idx = sorted(indices)
    best_start = current_start = sorted_idx[0]
    best_end = current_end = sorted_idx[0]
    best_length = 1
    current_length = 1
    for idx in sorted_idx[1:]:
        if idx == current_end + 1:
            current_end = idx
            current_length += 1
        else:
            if current_length > best_length:
                best_start, best_end = current_start, current_end
                best_length = current_length
            current_start = current_end = idx
            current_length = 1
    if current_length > best_length:
        best_start, best_end = current_start, current_end
    return best_start, best_end


def integrate_signals(
    cosine: CosineCurves,
    parameter_layers: Sequence[int],
) -> KeyLayerIntervals:
    """Combine representational and parameter signals to determine key layers."""

    rep_interval = (cosine.k_start, max(cosine.k_start, cosine.k_end_rep))
    filtered = [
        layer
        for layer in parameter_layers
        if rep_interval[0] <= layer <= rep_interval[1]
    ]
    key_interval = _longest_consecutive_segment(filtered)
    return KeyLayerIntervals(representational=rep_interval, key=key_interval)


def identify_key_layers(
    normal_states: Sequence[torch.Tensor],
    risk_states: Sequence[torch.Tensor],
    base_state: Dict[str, torch.Tensor],
    defense_state: Dict[str, torch.Tensor],
    *,
    num_pairs: int = 500,
    seed: Optional[int] = None,
    smoothing: int = 5,
    baseline_layers: int = 3,
    z_threshold: float = 1.0,
) -> KeyLayerAnalysisResult:
    """Full pipeline for computing key-layer intervals from activations and parameters."""

    cosine = compute_cosine_curves(
        normal_states,
        risk_states,
        num_pairs=num_pairs,
        seed=seed,
        smoothing=smoothing,
        baseline_layers=baseline_layers,
    )
    num_layers = len(normal_states)
    parameter_curves, significant_layers = compute_parameter_curves(
        base_state,
        defense_state,
        num_layers=num_layers,
        z_threshold=z_threshold,
    )
    intervals = integrate_signals(cosine, significant_layers)
    return KeyLayerAnalysisResult(cosine=cosine, parameters=parameter_curves, intervals=intervals)

