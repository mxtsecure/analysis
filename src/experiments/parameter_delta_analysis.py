"""CLI for running granular parameter delta analysis between two checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.parameter_deltas import (
    ParameterDeltaSummary,
    load_state_dict,
    summarise_parameter_deltas,
)
from analysis.visualization import export_summary_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Path to the reference checkpoint")
    parser.add_argument("--finetuned-model", required=True, help="Path to the finetuned checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for CSVs and plots")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default=None,
        help="Optional torch dtype to use when loading checkpoints",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to load the checkpoints on before materialising CPU tensors",
    )
    parser.add_argument(
        "--heatmap-metrics",
        nargs="*",
        default=("mean_abs", "mean_abs_rel"),
        help="Metrics to render as heatmaps",
    )
    parser.add_argument(
        "--curve-metrics",
        nargs="*",
        default=("mean_abs", "mean_l2", "mean_abs_rel"),
        help="Metrics to plot as layer-wise curves",
    )
    parser.add_argument(
        "--export-json",
        action="store_true",
        help="Also export the summary as JSON for downstream scripting",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip rendering plots and only export CSV tables",
    )
    return parser.parse_args()


def _resolve_dtype(label: Optional[str]) -> Optional[torch.dtype]:
    if label is None:
        return None
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[label]


def _state_dict_from_checkpoint(path: str, *, torch_dtype: Optional[torch.dtype], device: str) -> dict:
    state = load_state_dict(path, torch_dtype=torch_dtype, device=device)
    return state


def run_analysis(args: argparse.Namespace) -> ParameterDeltaSummary:
    dtype = _resolve_dtype(args.dtype)
    base_state = _state_dict_from_checkpoint(args.base_model, torch_dtype=dtype, device=args.device)
    finetuned_state = _state_dict_from_checkpoint(
        args.finetuned_model, torch_dtype=dtype, device=args.device
    )
    return summarise_parameter_deltas(base_state, finetuned_state)


def main() -> None:  # pragma: no cover - CLI entry point
    args = parse_args()
    summary = run_analysis(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_plots:
        artifacts = export_summary_artifacts(
            summary,
            args.output_dir,
            heatmap_metrics=(),
            curve_metrics=(),
        )
        print("Plots skipped; CSVs written to output directory.")
    else:
        artifacts = export_summary_artifacts(
            summary,
            args.output_dir,
            heatmap_metrics=args.heatmap_metrics,
            curve_metrics=args.curve_metrics,
        )
        print("Saved artifacts:")

    if args.export_json:
        serialisable = {
            "parameters": summary.parameters.to_dict(orient="records"),
            "layers": summary.layers.to_dict(orient="records"),
            "modules": summary.modules.to_dict(orient="records"),
            "submodules": summary.submodules.to_dict(orient="records"),
        }
        json_path = args.output_dir / "summary.json"
        json_path.write_text(json.dumps(serialisable, indent=2))
        artifacts["summary_json"] = json_path

    for label, path in sorted(artifacts.items()):
        print(f"  {label}: {path}")


if __name__ == "__main__":  # pragma: no cover
    main()
