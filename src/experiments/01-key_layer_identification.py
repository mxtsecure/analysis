"""CLI for identifying key representation layers of a defense model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.key_layers import (
    KeyLayerAnalysisResult,
    collect_last_token_hidden_states,
    identify_key_layers,
)
from analysis.visualization import plot_key_layer_analysis
from data.datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/finetune/gemma-2-2b-it-tofu", help="Path to M_tofu (base model)") 
    parser.add_argument("--defense-model", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/unlearn/gemma-2-2b-it-tofu/gemma-2-2b-it-tofu-NPO", help="Path to the defense model") 
    parser.add_argument("--normal", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/normal.jsonl", type=Path, help="Normal query dataset path") 
    parser.add_argument("--risk", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/safety.jsonl", type=Path, help="Risk query dataset path") 
    parser.add_argument("--output", default="/data/xiangtao/projects/crossdefense/code/analysis_results/0-key_layers/gemma-2-2b-it-tofu-NPO/result.json", type=Path, help="Output JSON file for metrics")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-pairs", type=int, default=500)
    parser.add_argument("--smoothing", type=int, default=5)
    parser.add_argument(
        "--baseline-percentile",
        type=float,
        default=30.0,
        help="Percentile used to select low-difference layers for baseline statistics",
    )
    parser.add_argument("--z-threshold", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Render the stacked cosine/angle/parameter plot alongside the JSON output.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default="/data/xiangtao/projects/crossdefense/code/analysis_results/0-key_layers/gemma-2-2b-it-tofu-NPO/result.png",
        help="Explicit path for saving the plot (implies --plot).",
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


def _load_tokenizer(model_path: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _create_dataloader(path: Path, tokenizer, batch_size: int, max_length: int) -> DataLoader:
    dataset = load_dataset(path, tokenizer=tokenizer, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _state_dict_cpu(model: AutoModelForCausalLM) -> dict:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def run_analysis(args: argparse.Namespace) -> KeyLayerAnalysisResult:
    dtype = _resolve_dtype(args.dtype)
    tokenizer = _load_tokenizer(args.defense_model)
    normal_loader = _create_dataloader(args.normal, tokenizer, args.batch_size, args.max_length)
    risk_loader = _create_dataloader(args.risk, tokenizer, args.batch_size, args.max_length)

    device = torch.device(args.device)
    defense_model = AutoModelForCausalLM.from_pretrained(args.defense_model, torch_dtype=dtype)
    defense_model.to(device)

    normal_states = collect_last_token_hidden_states(defense_model, normal_loader, device)
    risk_states = collect_last_token_hidden_states(defense_model, risk_loader, device)

    # Move model back to CPU before allocating the base model for diffing.
    defense_model.cpu()
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)

    base_state = _state_dict_cpu(base_model)
    defense_state = _state_dict_cpu(defense_model)

    return identify_key_layers(
        normal_states=normal_states,
        risk_states=risk_states,
        base_state=base_state,
        defense_state=defense_state,
        num_pairs=args.num_pairs,
        seed=args.seed,
        smoothing=args.smoothing,
        baseline_percentile=args.baseline_percentile,
        z_threshold=args.z_threshold,
    )


def main() -> None:  # pragma: no cover - CLI entry point
    args = parse_args()
    result = run_analysis(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result.to_dict(), indent=2))

    print("=== Key Layer Analysis Summary ===")   
    
    # --- 1. 表示层关键分歧区间 ---
    rep_spans = result.intervals.representational
    if rep_spans:
        spans_str = ", ".join(f"({start}, {end})" for start, end in rep_spans)
        print(f"Representational Spans (Merged): [{spans_str}]")
    else:
        print("Representational Spans: None detected or merged.")
    
    # --- 2. 参数层显著区间 ---
    param_spans = result.intervals.parameter_spans
    if param_spans:
        spans_str = ", ".join(
            f"({start}, {end})" for start, end in param_spans
        )
        print(f"Parameter-Significant Spans: [{spans_str}]")
    else:
        print("Parameter-Significant Spans: None (Did not exceed z-threshold).")

    # --- 3. 信号重叠区间 ---
    overlap_segments = []
    for rep_start, rep_end in rep_spans:
        for span_start, span_end in param_spans:
            overlap_start = max(rep_start, span_start)
            overlap_end = min(rep_end, span_end)
            
            if overlap_start <= overlap_end:
                overlap_segments.append((overlap_start, overlap_end))

    if overlap_segments:
        overlap_str = ", ".join(
            f"({start}, {end})" for start, end in overlap_segments
        )
        print(f"Overlap between Signals: [{overlap_str}]")
    else:
        print("Overlap between Signals: None detected.")

    # --- 4. 绘图和输出 ---  
    if args.plot or args.plot_path is not None:
        plot_path = args.plot_path or args.output.with_suffix(".png")
        
        critical_layers = set()
        for start, end in rep_spans:
             critical_layers.update(range(start, end + 1)) 
          
        plot_key_layer_analysis(
            result,
            plot_path,
            title=f"Key-layer analysis: {Path(args.defense_model).name}",
            critical_layers=sorted(list(critical_layers)),
        )
        print(f"Plot saved to {plot_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

