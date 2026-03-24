"""CLI for identifying key representation layers comparing two defense models with distinct datasets."""
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
# 确保路径包含项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.key_layers import (
    CosineCurves,
    KeyLayerAnalysisResult,
    collect_last_token_hidden_states,
    compute_baseline_adjusted_cosine,
    compute_cosine_curves,
    identify_key_layers,
    integrate_signals,
)
from analysis.visualization import plot_dual_key_layer_analysis
from data.datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    
    # --- Base Model ---
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-1B-Instruct", help="Path to base model (e.g. M_tofu)")
    
    # --- Defense Model 1 Configuration ---
    parser.add_argument("--defense-model-1", default="/data/xiangtao/projects/crossdefense/code/defense/safety/DPO/DPO_models/Llama-3.2-1B-Instruct-tofu-DPO", help="Path to first defense model")
    parser.add_argument("--title-1", default="Safety Defense (DPO)", help="Plot title for first model")
    parser.add_argument("--normal-1", default="/data/xiangtao/projects/crossdefense/code/mechanistic_analysis/datasets/risk_data/normal.jsonl", type=Path, help="Normal query dataset for Model 1")
    parser.add_argument("--risk-1", default="/data/xiangtao/projects/crossdefense/code/mechanistic_analysis/datasets/risk_data/safety.jsonl", type=Path, help="Risk query dataset for Model 1")
    
    # --- Defense Model 2 Configuration ---
    parser.add_argument("--defense-model-2", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/unlearn/Llama-3.2-1B-Instruct-tofu-DPO-NPO", help="Path to second defense model")
    parser.add_argument("--title-2", default="Privacy Defense (NPO)", help="Plot title for second model")
    parser.add_argument("--normal-2", default="/data/xiangtao/projects/crossdefense/code/mechanistic_analysis/datasets/risk_data/normal.jsonl", type=Path, help="Normal query dataset for Model 2")
    parser.add_argument("--risk-2", default="/data/xiangtao/projects/crossdefense/code/mechanistic_analysis/datasets/risk_data/privacy_risk.jsonl", type=Path, help="Risk query dataset for Model 2")

    # --- Output & Analysis Settings ---
    parser.add_argument("--output-dir", default="/data/xiangtao/projects/crossdefense/code/mechanistic_analysis/results/01-key_layers/Llama-3.2-1B-Instruct-tofu-DPO-NPO-test", type=Path, help="Directory to save jsons and plot")
    parser.add_argument("--plot-name", default="comparison_plot.pdf", type=str, help="Filename for the comparison plot")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate plot from existing result_model_1.json and result_model_2.json")
    parser.add_argument("--no-baseline-adjust", action="store_true", help="Disable baseline adjustment (key layers = raw defense only; default is to use base-model baseline for defense-induced key layers)")
    
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-pairs", type=int, default=500)
    parser.add_argument("--smoothing", type=int, default=5)
    parser.add_argument("--baseline-percentile", type=float, default=30.0)
    parser.add_argument("--z-threshold", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    
    return parser.parse_args()


def _resolve_dtype(label: Optional[str]) -> Optional[torch.dtype]:
    if label is None: return None
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
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


def compute_baseline_cosine(
    base_model: AutoModelForCausalLM,
    normal_path: Path,
    risk_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> CosineCurves:
    """在基座模型上跑表示分析，得到 N-N/N-R 角差曲线，用于后续「防御−基线」调整。"""
    tokenizer = _load_tokenizer(args.base_model)
    normal_loader = _create_dataloader(normal_path, tokenizer, args.batch_size, args.max_length)
    risk_loader = _create_dataloader(risk_path, tokenizer, args.batch_size, args.max_length)
    normal_states = collect_last_token_hidden_states(base_model, normal_loader, device)
    risk_states = collect_last_token_hidden_states(base_model, risk_loader, device)
    return compute_cosine_curves(
        normal_states,
        risk_states,
        num_pairs=args.num_pairs,
        seed=args.seed,
        smoothing=args.smoothing,
        baseline_percentile=args.baseline_percentile,
    )


def compute_model_cosine_from_path(
    model_path: str,
    normal_path: Path,
    risk_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> CosineCurves:
    """通用版本：给定任意模型路径，计算其在指定 normal/risk 上的表示角差曲线。

    用于 second defense 相对 first defense 的基线（即 defense-model-2 与 defense-model-1 对比）。
    """
    print(f"Loading model for baseline representation: {Path(model_path).name}")
    dtype = _resolve_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(device)

    tokenizer = _load_tokenizer(model_path)
    normal_loader = _create_dataloader(normal_path, tokenizer, args.batch_size, args.max_length)
    risk_loader = _create_dataloader(risk_path, tokenizer, args.batch_size, args.max_length)

    normal_states = collect_last_token_hidden_states(model, normal_loader, device)
    risk_states = collect_last_token_hidden_states(model, risk_loader, device)

    model.cpu()
    del model
    torch.cuda.empty_cache()

    return compute_cosine_curves(
        normal_states,
        risk_states,
        num_pairs=args.num_pairs,
        seed=args.seed,
        smoothing=args.smoothing,
        baseline_percentile=args.baseline_percentile,
    )


def analyze_single_model(
    model_path: str,
    normal_path: Path,  # 新增：接收具体路径
    risk_path: Path,    # 新增：接收具体路径
    base_state: dict,
    args: argparse.Namespace
) -> KeyLayerAnalysisResult:
    """运行单个模型的分析逻辑，使用指定的 Normal/Risk 数据集"""
    print(f"\n--- Analyzing Model: {Path(model_path).name} ---")
    print(f"    Normal Data: {normal_path}")
    print(f"    Risk Data:   {risk_path}")
    
    dtype = _resolve_dtype(args.dtype)
    device = torch.device(args.device)
    
    tokenizer = _load_tokenizer(model_path)
    
    # 使用传入的 specific path 创建 DataLoader
    normal_loader = _create_dataloader(normal_path, tokenizer, args.batch_size, args.max_length)
    risk_loader = _create_dataloader(risk_path, tokenizer, args.batch_size, args.max_length)

    print("Loading defense model...")
    defense_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    defense_model.to(device)

    print("Collecting hidden states...")
    normal_states = collect_last_token_hidden_states(defense_model, normal_loader, device)
    risk_states = collect_last_token_hidden_states(defense_model, risk_loader, device)

    defense_model.cpu()
    defense_state = _state_dict_cpu(defense_model)
    
    # 清理显存
    del defense_model
    torch.cuda.empty_cache()

    print("Computing metrics...")
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
    ), defense_state


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        # 仅重新生成图，但仍会重新跑一遍前向（不写出 JSON）
        dtype = _resolve_dtype(args.dtype)
        device = torch.device(args.device)

        print(f"Loading Base Model: {args.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
        base_model.to(device)
        base_state = _state_dict_cpu(base_model)

        print("\n--- Base model representation (for angular gaps) ---")
        print("    Baseline 1: same data as Model 1")
        baseline_cosine_1 = compute_baseline_cosine(
            base_model, args.normal_1, args.risk_1, args, device
        )
        print("    Baseline 2: Model 2 vs Defense Model 1 (same data as Model 2)")
        baseline_cosine_2 = compute_model_cosine_from_path(
            args.defense_model_1, args.normal_2, args.risk_2, args, device
        )
        base_model.cpu()
        del base_model
        torch.cuda.empty_cache()

        # 防御模型 1
        result1_raw, defense_state1 = analyze_single_model(
            model_path=args.defense_model_1,
            normal_path=args.normal_1,
            risk_path=args.risk_1,
            base_state=base_state,
            args=args,
        )

        # 防御模型 2（顺序部署语义：基线是 defense-model-1）
        result2_raw, _ = analyze_single_model(
            model_path=args.defense_model_2,
            normal_path=args.normal_2,
            risk_path=args.risk_2,
            base_state=defense_state1,
            args=args,
        )

        print("\nGenerating Comparison Plot (angular gaps)...")
        plot_path = args.output_dir / args.plot_name
        plot_dual_key_layer_analysis(
            baseline1=baseline_cosine_1,
            defense1=result1_raw.cosine,
            name1=args.title_1,
            baseline2=baseline_cosine_2,
            defense2=result2_raw.cosine,
            name2=args.title_2,
            output_path=plot_path,
            figsize=(5, 3),
        )
        print(f"Plot saved to {plot_path}")
        return

    dtype = _resolve_dtype(args.dtype)
    device = torch.device(args.device)

    # 1. Load Base Model
    print(f"Loading Base Model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    base_model.to(device)
    base_state = _state_dict_cpu(base_model)

    print("\n--- Base model representation (for angular gaps / baseline-adjustment) ---")
    print("    Baseline 1: same data as Model 1")
    baseline_cosine_1 = compute_baseline_cosine(
        base_model, args.normal_1, args.risk_1, args, device
    )
    print("    Baseline 2: Model 2 vs Defense Model 1 (same data as Model 2)")
    # 对于第二个防御模型，我们关心的是它在 first defense 之上的增量效果，
    # 因此这里用 defense-model-1 作为表示基线，而不是基座模型。
    baseline_cosine_2 = compute_model_cosine_from_path(
        args.defense_model_1, args.normal_2, args.risk_2, args, device
    )
    base_model.cpu()
    del base_model
    torch.cuda.empty_cache()

    # 2. 分析模型 1 (使用 dataset 1)
    result1_raw, defense_state1 = analyze_single_model(
        model_path=args.defense_model_1,
        normal_path=args.normal_1,
        risk_path=args.risk_1,
        base_state=base_state,
        args=args,
    )
    if not args.no_baseline_adjust:
        adj_cos_1 = compute_baseline_adjusted_cosine(result1_raw.cosine, baseline_cosine_1)
        result1 = KeyLayerAnalysisResult(
            cosine=adj_cos_1,
            parameters=result1_raw.parameters,
            intervals=integrate_signals(adj_cos_1, result1_raw.intervals.parameter_layers),
        )
    else:
        result1 = result1_raw
    (args.output_dir / "result_model_1.json").write_text(json.dumps(result1.to_dict(), indent=2))

    # 3. 分析模型 2 (使用 dataset 2)
    result2_raw, _ = analyze_single_model(
        model_path=args.defense_model_2,
        normal_path=args.normal_2,
        risk_path=args.risk_2,
        base_state=defense_state1,
        args=args,
    )
    if not args.no_baseline_adjust:
        adj_cos_2 = compute_baseline_adjusted_cosine(result2_raw.cosine, baseline_cosine_2)
        result2 = KeyLayerAnalysisResult(
            cosine=adj_cos_2,
            parameters=result2_raw.parameters,
            intervals=integrate_signals(adj_cos_2, result2_raw.intervals.parameter_layers),
        )
    else:
        result2 = result2_raw
    (args.output_dir / "result_model_2.json").write_text(json.dumps(result2.to_dict(), indent=2))

    # 4. 绘图 (合并结果)
    print("\nGenerating Comparison Plot (angular gaps)...")
    plot_path = args.output_dir / args.plot_name
    plot_dual_key_layer_analysis(
        baseline1=baseline_cosine_1,
        defense1=result1_raw.cosine,
        name1=args.title_1,
        baseline2=baseline_cosine_2,
        defense2=result2_raw.cosine,
        name2=args.title_2,
        output_path=plot_path,
        figsize=(5, 3),
    )

if __name__ == "__main__":
    main()