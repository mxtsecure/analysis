"""CLI for running data-oblivious critical layer analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.data_oblivious_layers import (
    DataObliviousCriticalLayerReport,
    identify_data_oblivious_critical_layers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/finetune/Llama-3.2-1B-Instruct-tofu",
        help="Path to the model checkpoint to analyse.",
    )
    parser.add_argument(
        "--output",
        default="/data/xiangtao/projects/crossdefense/code/analysis_results/0-key_layers/Llama-3.2-1B-Instruct-tofu-DPO/data_oblivious.json",
        type=Path,
        help="Output JSON file for critical layer statistics.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Synthetic batch size for probing.")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length of synthetic prompts (defaults to model max length).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=32,
        help="Number of random batches used to estimate saliency statistics.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override vocabulary size used for random token sampling.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="Model dtype used for analysis.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the analysis on.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top ranked layers to display in the console summary.",
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


def run_analysis(args: argparse.Namespace) -> DataObliviousCriticalLayerReport:
    dtype = _resolve_dtype(args.dtype)
    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)

    report = identify_data_oblivious_critical_layers(
        model,
        device=device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_batches=args.num_batches,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    model.cpu()
    return report


def main() -> None:  # pragma: no cover - CLI entry point
    args = parse_args()
    report = run_analysis(args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report.to_dict(), indent=2))

    print("=== Data-Oblivious Critical Layer Summary ===")
    print(f"Model: {args.model}")
    print(f"Output saved to: {args.output}")

    top_layers = report.topk(args.topk)
    if top_layers:
        print("Top-ranked layers (normalized scores):")
        for idx, (name, score) in enumerate(top_layers, start=1):
            print(f"  {idx:>2}: {name} -> {score:.4f}")
    else:
        print("No layers were scored.")


if __name__ == "__main__":  # pragma: no cover
    main()
