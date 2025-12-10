"""Quantify how the second defense's weight changes project onto the first."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional, Tuple

import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.parameter_deltas import load_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Path to the base (pre-defense) checkpoint")
    parser.add_argument("--defense1", required=True, help="Path to the first defense checkpoint")
    parser.add_argument("--defense2", required=True, help="Path to the second defense checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/data/xiangtao/projects/crossdefense/code/analysis_results/10-defense_weight_projection/metrics.json"),
        help="Where to save the projection metrics (JSON)",
    )
    parser.add_argument("--device", default="cpu", help="Device used when loading checkpoints")
    parser.add_argument("--torch-dtype", default=None, help="Optional torch dtype override (e.g., float16)")
    parser.add_argument("--eps", type=float, default=1e-12, help="Small constant to avoid division by zero")
    return parser.parse_args()


def _extract_layer_index(parameter_name: str) -> Optional[int]:
    parts = parameter_name.split(".")
    for idx in range(len(parts) - 1):
        if parts[idx] == "layers" and idx > 0 and parts[idx - 1] == "model":
            if idx + 1 >= len(parts):
                break
            layer_id = parts[idx + 1]
            if not layer_id.isdigit():
                break
            return int(layer_id)
    return None

def _accumulate_projection(
    base: Mapping[str, torch.Tensor],
    defense1: Mapping[str, torch.Tensor],
    defense2: Mapping[str, torch.Tensor],
    *,
    eps: float,
) -> Tuple[MutableMapping[str, float], Dict[int, MutableMapping[str, float]]]:
    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0
    per_layer: Dict[int, MutableMapping[str, float]] = {}

    for name in sorted(set(base.keys()) & set(defense1.keys()) & set(defense2.keys())):
        delta1 = (defense1[name] - base[name]).float()
        delta2 = (defense2[name] - defense1[name]).float()
        flat_delta1 = delta1.reshape(-1)
        flat_delta2 = delta2.reshape(-1)
        dot_contrib = float(torch.dot(flat_delta1, flat_delta2).item())
        dot += dot_contrib
        norm1_sq += float(flat_delta1.pow(2).sum().item())
        norm2_sq += float(flat_delta2.pow(2).sum().item())

        layer_idx = _extract_layer_index(name)
        if layer_idx is None:
            continue
        stats = per_layer.setdefault(layer_idx, {"dot": 0.0, "norm1_sq": 0.0, "norm2_sq": 0.0})
        stats["dot"] += dot_contrib
        stats["norm1_sq"] += float(flat_delta1.pow(2).sum().item())
        stats["norm2_sq"] += float(flat_delta2.pow(2).sum().item())

    global_stats: MutableMapping[str, float] = {
        "dot": dot,
        "delta1_norm": norm1_sq**0.5,
        "delta2_norm": norm2_sq**0.5,
    }

    def _finalise(record: MutableMapping[str, float]) -> None:
        record["cosine"] = record["dot"] / ((record.get("delta1_norm", 0.0) * record.get("delta2_norm", 0.0)) + eps)
        record["scalar_projection"] = record["dot"] / (record.get("delta1_norm", 0.0) + eps)
        record["projection_coefficient"] = record["dot"] / ((record.get("delta1_norm", 0.0) ** 2) + eps)

    _finalise(global_stats)
    for stats in per_layer.values():
        stats["delta1_norm"] = stats["norm1_sq"]**0.5
        stats["delta2_norm"] = stats["norm2_sq"]**0.5
        _finalise(stats)
        stats.pop("norm1_sq", None)
        stats.pop("norm2_sq", None)

    return global_stats, per_layer


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.torch_dtype) if args.torch_dtype else None

    base_state = load_state_dict(args.base, torch_dtype=dtype, device=args.device)
    defense1_state = load_state_dict(args.defense1, torch_dtype=dtype, device=args.device)
    defense2_state = load_state_dict(args.defense2, torch_dtype=dtype, device=args.device)

    global_stats, per_layer = _accumulate_projection(
        base_state, defense1_state, defense2_state, eps=args.eps
    )

    metrics = {
        "interaction": global_stats,
        "per_layer": [
            {"layer": layer, **stats} for layer, stats in sorted(per_layer.items(), key=lambda item: item[0])
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
