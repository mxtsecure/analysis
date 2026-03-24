"""Utilities for identifying key representation layers for safety/privacy defenses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class CosineCriticalInterval:
    """Represents a contiguous segment where angular differences spike."""

    start: int
    peak: int
    end: int


@dataclass
class CosineCurves:
    """Container storing cosine-based statistics for each transformer layer."""

    mu_nn: List[float]
    mu_nr: List[float]
    std_nn: List[float]
    std_nr: List[float]
    angle_nn: List[float]
    angle_nr: List[float]
    angle_std_nn: List[float]
    angle_std_nr: List[float]
    delta_phi: List[float]
    critical_intervals: List[CosineCriticalInterval]


@dataclass
class ParameterCurves:
    """Stores mean absolute parameter deltas for attention/MLP blocks by layer."""

    s_attn: List[float]
    s_mlp: List[float]
    s_total: List[float]
    s_zscore: List[float]


@dataclass
class KeyLayerIntervals:
    """Captures representation- and parameter-driven layer signals independently."""

    representational: List[Tuple[int, int]]
    parameter_layers: List[int]
    parameter_spans: List[Tuple[int, int]]


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
                "angle_std_nn": self.cosine.angle_std_nn,
                "angle_std_nr": self.cosine.angle_std_nr,
                "delta_phi": self.cosine.delta_phi,
                "critical_intervals": [
                    {"start": interval.start, "peak": interval.peak, "end": interval.end}
                    for interval in self.cosine.critical_intervals
                ],
            },
            "parameters": {
                "s_attn": self.parameters.s_attn,
                "s_mlp": self.parameters.s_mlp,
                "s_total": self.parameters.s_total,
                "s_zscore": self.parameters.s_zscore,
            },
            "intervals": {
                "representational": list(self.intervals.representational),
                "parameter_layers": list(self.intervals.parameter_layers),
                "parameter_spans": [
                    [start, end] for start, end in self.intervals.parameter_spans
                ],
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> KeyLayerAnalysisResult:
        """Reconstruct result from a JSON-serialised dictionary (e.g. from result_model_*.json)."""
        c = data["cosine"]
        intervals_raw = c.get("critical_intervals", [])
        # 兼容旧版本：angle_std_* 可能不存在
        angle_std_nn = c.get("angle_std_nn")
        angle_std_nr = c.get("angle_std_nr")
        if angle_std_nn is None:
            angle_std_nn = [0.0 for _ in c["angle_nn"]]
        if angle_std_nr is None:
            angle_std_nr = [0.0 for _ in c["angle_nr"]]
        cosine = CosineCurves(
            mu_nn=c["mu_nn"],
            mu_nr=c["mu_nr"],
            std_nn=c["std_nn"],
            std_nr=c["std_nr"],
            angle_nn=c["angle_nn"],
            angle_nr=c["angle_nr"],
            angle_std_nn=angle_std_nn,
            angle_std_nr=angle_std_nr,
            delta_phi=c["delta_phi"],
            critical_intervals=[
                CosineCriticalInterval(start=i["start"], peak=i["peak"], end=i["end"])
                for i in intervals_raw
            ],
        )
        p = data["parameters"]
        parameters = ParameterCurves(
            s_attn=p["s_attn"],
            s_mlp=p["s_mlp"],
            s_total=p["s_total"],
            s_zscore=p["s_zscore"],
        )
        i = data["intervals"]
        intervals = KeyLayerIntervals(
            representational=[tuple(t) for t in i["representational"]],
            parameter_layers=list(i["parameter_layers"]),
            parameter_spans=[tuple(t) for t in i["parameter_spans"]],
        )
        return cls(cosine=cosine, parameters=parameters, intervals=intervals)


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
    baseline_percentile: float = 30.0,
) -> CosineCurves:
    """Compute cosine/angle statistics for N–N and N–R comparisons.

    Instead of relying on onset/offset thresholds, we highlight every segment that shows a
    rising trend in the smoothed angular differences so downstream consumers can track all
    regions where divergence increases.
    """

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
    angle_std_nn = []
    angle_std_nr = []

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
        ang_nn_vals = torch.rad2deg(torch.acos(clamped_nn))
        ang_nr_vals = torch.rad2deg(torch.acos(clamped_nr))
        ang_nn_mean = ang_nn_vals.mean().item()
        ang_nr_mean = ang_nr_vals.mean().item()
        angle_nn.append(ang_nn_mean)
        angle_nr.append(ang_nr_mean)
        angle_std_nn.append(ang_nn_vals.std(unbiased=False).item())
        angle_std_nr.append(ang_nr_vals.std(unbiased=False).item())
        cos_delta.append(ang_nr_mean - ang_nn_mean)

    smooth_delta = np.array(cos_delta)
    positive_slope = np.diff(smooth_delta) > 0.0
    critical_intervals: List[CosineCriticalInterval] = []
    idx = 0
    while idx < len(positive_slope):
        if not positive_slope[idx]:
            idx += 1
            continue
        seg_start = idx
        while idx < len(positive_slope) and positive_slope[idx]:
            idx += 1
        seg_end = idx  # idx already points to the first non-positive slope (exclusive)
        interval_start = seg_start
        interval_end = seg_end
        segment = smooth_delta[interval_start : interval_end + 1]
        peak_offset = int(np.argmax(segment))
        peak_idx = interval_start + peak_offset
        critical_intervals.append(
            CosineCriticalInterval(
                start=int(interval_start), peak=int(peak_idx), end=int(interval_end)
            )
        )

    return CosineCurves(
        mu_nn=mu_nn,
        mu_nr=mu_nr,
        std_nn=std_nn,
        std_nr=std_nr,
        angle_nn=angle_nn,
        angle_nr=angle_nr,
        angle_std_nn=angle_std_nn,
        angle_std_nr=angle_std_nr,
        delta_phi=cos_delta,
        critical_intervals=critical_intervals,
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

def _critical_intervals_from_delta_phi(delta_phi: Sequence[float]) -> List[CosineCriticalInterval]:
    """从每层的角差序列中找出上升段，得到关键区间（与 compute_cosine_curves 中逻辑一致）。"""
    if not delta_phi:
        return []
    smooth_delta = np.array(delta_phi)
    positive_slope = np.diff(smooth_delta) > 0.0
    critical_intervals: List[CosineCriticalInterval] = []
    idx = 0
    while idx < len(positive_slope):
        if not positive_slope[idx]:
            idx += 1
            continue
        seg_start = idx
        while idx < len(positive_slope) and positive_slope[idx]:
            idx += 1
        seg_end = idx
        interval_start = seg_start
        interval_end = seg_end
        segment = smooth_delta[interval_start : interval_end + 1]
        peak_offset = int(np.argmax(segment))
        peak_idx = interval_start + peak_offset
        critical_intervals.append(
            CosineCriticalInterval(
                start=int(interval_start), peak=int(peak_idx), end=int(interval_end)
            )
        )
    return critical_intervals


def compute_baseline_adjusted_cosine(
    defense_cosine: CosineCurves,
    baseline_cosine: CosineCurves,
) -> CosineCurves:
    """用「防御 - 基线」的角差得到基线调整后的余弦曲线，关键区间基于增量角差。

    这样得到的关键层反映的是防御在基座之上新增的分离（而非基座已有的过增强差异）。
    """
    if len(defense_cosine.delta_phi) != len(baseline_cosine.delta_phi):
        raise ValueError(
            "defense_cosine and baseline_cosine must have same number of layers"
        )
    delta_phi_adjusted = [
        d - b for d, b in zip(defense_cosine.delta_phi, baseline_cosine.delta_phi)
    ]
    critical_intervals = _critical_intervals_from_delta_phi(delta_phi_adjusted)
    return CosineCurves(
        mu_nn=defense_cosine.mu_nn,
        mu_nr=defense_cosine.mu_nr,
        std_nn=defense_cosine.std_nn,
        std_nr=defense_cosine.std_nr,
        angle_nn=defense_cosine.angle_nn,
        angle_nr=defense_cosine.angle_nr,
        angle_std_nn=defense_cosine.angle_std_nn,
        angle_std_nr=defense_cosine.angle_std_nr,
        delta_phi=delta_phi_adjusted,
        critical_intervals=critical_intervals,
    )


def _merge_intervals(intervals: Sequence[CosineCriticalInterval]) -> List[Tuple[int, int]]:
    """Merges a list of CosineCriticalIntervals into a minimal list of non-overlapping spans (start, end)."""
    if not intervals:
        return []
    spans = sorted([(i.start, i.end) for i in intervals], key=lambda x: x[0])
    
    merged = []
    current_start, current_end = spans[0]

    for next_start, next_end in spans[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    merged.append((current_start, current_end))
    return merged

def _group_consecutive_layers(layers: Sequence[int]) -> List[Tuple[int, int]]:
    if not layers:
        return []
    sorted_layers = sorted(layers)
    groups: List[Tuple[int, int]] = []
    start = prev = sorted_layers[0]
    for layer in sorted_layers[1:]:
        if layer == prev + 1:
            prev = layer
        else:
            groups.append((start, prev))
            start = prev = layer
    groups.append((start, prev))
    return groups


def integrate_signals(
    cosine: CosineCurves,
    parameter_layers: Sequence[int],
) -> KeyLayerIntervals:
    """Summarise representational and parameter signals without intersecting them."""

    rep_interval = _merge_intervals(cosine.critical_intervals)
    unique_layers = sorted(set(parameter_layers))
    parameter_spans = _group_consecutive_layers(unique_layers)
    return KeyLayerIntervals(
        representational=rep_interval,
        parameter_layers=unique_layers,
        parameter_spans=parameter_spans,
    )


def identify_key_layers(
    normal_states: Sequence[torch.Tensor],
    risk_states: Sequence[torch.Tensor],
    base_state: Dict[str, torch.Tensor],
    defense_state: Dict[str, torch.Tensor],
    *,
    num_pairs: int = 500,
    seed: Optional[int] = None,
    smoothing: int = 5,
    baseline_percentile: float = 30.0,
    z_threshold: float = 1.0,
) -> KeyLayerAnalysisResult:
    """Full pipeline for computing key-layer intervals from activations and parameters."""

    cosine = compute_cosine_curves(
        normal_states,
        risk_states,
        num_pairs=num_pairs,
        seed=seed,
        smoothing=smoothing,
        baseline_percentile=baseline_percentile,
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

