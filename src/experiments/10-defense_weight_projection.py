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

DEFAULT_CONFIG_PATH = Path("/data/xiangtao/projects/crossdefense/code/mechanistic_analysis/src/configs/privacy_defense_pairs.json")
DEFAULT_OUTPUT_ROOT = Path("/data/xiangtao/projects/crossdefense/code/mechanistic_analysis/results/10-old/privacy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=None, help="Path to the base (pre-defense) checkpoint")
    parser.add_argument("--defense1", default=None, help="Path to the first defense checkpoint")
    parser.add_argument("--defense2", default=None, help="Path to the second defense checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        default="/data/xiangtao/projects/crossdefense/code/mechanistic_analysis/results/10-old/privacy",
        help="Where to save the projection metrics (JSON). If using --config, this is ignored.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default="/data/xiangtao/projects/crossdefense/code/mechanistic_analysis/src/configs/privacy_defense_pairs.json",
        help=f"Path to JSON config file with experiment pairs. If provided, will process all experiments in the config. Default: {DEFAULT_CONFIG_PATH}",
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


def process_single_experiment(
    base_path: str,
    defense1_path: str,
    defense2_path: str,
    output_path: Path,
    device: str,
    torch_dtype: Optional[str],
    eps: float,
) -> None:
    """处理单个实验的投影计算."""
    print(f"\n处理实验: base={base_path}, defense1={defense1_path}, defense2={defense2_path}")
    dtype = getattr(torch, torch_dtype) if torch_dtype else None

    base_state = load_state_dict(base_path, torch_dtype=dtype, device=device)
    defense1_state = load_state_dict(defense1_path, torch_dtype=dtype, device=device)
    defense2_state = load_state_dict(defense2_path, torch_dtype=dtype, device=device)

    global_stats, per_layer = _accumulate_projection(
        base_state, defense1_state, defense2_state, eps=eps
    )

    metrics = {
        "interaction": global_stats,
        "per_layer": [
            {"layer": layer, **stats} for layer, stats in sorted(per_layer.items(), key=lambda item: item[0])
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"结果已保存到: {output_path}")
    print(f"全局统计: cosine={global_stats['cosine']:.6f}, scalar_projection={global_stats['scalar_projection']:.6f}")


def main() -> None:
    args = parse_args()
    
    # 判断使用批量模式还是单个实验模式
    use_batch_mode = False
    
    if args.config is not None:
        # 用户明确指定了配置文件
        config_path = args.config
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        use_batch_mode = True
    elif args.base is None and args.defense1 is None and args.defense2 is None:
        # 用户没有提供单个实验参数，尝试使用默认配置文件
        config_path = DEFAULT_CONFIG_PATH
        if config_path.exists():
            use_batch_mode = True
        else:
            raise ValueError(
                f"未提供实验参数且默认配置文件不存在: {config_path}\n"
                "请提供 --base, --defense1, --defense2 参数，或使用 --config 指定配置文件"
            )
    
    if use_batch_mode:
        # 从配置文件读取并批量处理
        print(f"从配置文件读取实验列表: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            experiments = json.load(f)
        
        print(f"找到 {len(experiments)} 个实验，开始批量处理...")
        for i, exp in enumerate(experiments, 1):
            exp_id = exp.get("id", f"exp{i}")
            base_path = exp["base"]
            defense1_path = exp["defense1"]
            defense2_path = exp["defense2"]
            
            # 生成输出路径：使用实验ID作为子目录名
            output_path = DEFAULT_OUTPUT_ROOT / exp_id / "metrics.json"
            
            try:
                process_single_experiment(
                    base_path=base_path,
                    defense1_path=defense1_path,
                    defense2_path=defense2_path,
                    output_path=output_path,
                    device=args.device,
                    torch_dtype=args.torch_dtype,
                    eps=args.eps,
                )
                print(f"[{i}/{len(experiments)}] 实验 {exp_id} 完成")
            except Exception as e:
                print(f"[{i}/{len(experiments)}] 实验 {exp_id} 失败: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n所有实验处理完成！")
    
    else:
        # 单个实验处理模式
        if args.base is None or args.defense1 is None or args.defense2 is None:
            parser = argparse.ArgumentParser(description=__doc__)
            parser.error("当不使用 --config 时，必须提供 --base, --defense1, 和 --defense2 参数")
        
        output_path = args.output
        if output_path is None:
            # 如果没有指定输出路径，从defense2路径生成
            defense2_name = Path(args.defense2).name
            output_path = DEFAULT_OUTPUT_ROOT / defense2_name / "metrics.json"
        
        process_single_experiment(
            base_path=args.base,
            defense1_path=args.defense1,
            defense2_path=args.defense2,
            output_path=output_path,
            device=args.device,
            torch_dtype=args.torch_dtype,
            eps=args.eps,
        )


if __name__ == "__main__":
    main()