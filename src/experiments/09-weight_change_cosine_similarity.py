"""计算 (defense1 - base LLM) 与 (defense2 - defense1) 权重参数变化的余弦相似度，逐层计算."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OUTPUT_ROOT = Path("/data/xiangtao/projects/crossdefense/code/analysis_results/09-weight_change_cosine_similarity")


def _extract_layer_index(name: str) -> Optional[int]:
    """从参数名称中提取层索引."""
    parts = name.split(".")
    for idx in range(len(parts) - 1):
        if parts[idx] == "layers" and idx > 0 and parts[idx - 1] == "model":
            if idx + 1 >= len(parts):
                break
            layer_id = parts[idx + 1]
            if layer_id.isdigit():
                return int(layer_id)
    return None


def _get_layer_parameters(
    state_dict: Dict[str, torch.Tensor],
    layer_idx: int,
) -> List[torch.Tensor]:
    """获取指定层的所有参数张量."""
    layer_params = []
    layer_prefix = f"model.layers.{layer_idx}."
    for name, tensor in state_dict.items():
        if name.startswith(layer_prefix):
            layer_params.append(tensor.flatten())
    return layer_params


def _compute_layer_weight_delta(
    base_state: Dict[str, torch.Tensor],
    defense_state: Dict[str, torch.Tensor],
    layer_idx: int,
) -> torch.Tensor:
    """计算指定层的权重差异向量."""
    base_params = _get_layer_parameters(base_state, layer_idx)
    defense_params = _get_layer_parameters(defense_state, layer_idx)
    
    if len(base_params) != len(defense_params):
        raise ValueError(f"Layer {layer_idx}: parameter count mismatch")
    
    deltas = []
    for base_param, defense_param in zip(base_params, defense_params):
        if base_param.shape != defense_param.shape:
            raise ValueError(f"Layer {layer_idx}: shape mismatch")
        deltas.append((defense_param - base_param).flatten())
    
    if not deltas:
        return torch.tensor([])
    return torch.cat(deltas)


def _compute_cosine_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor, eps: float = 1e-12) -> float:
    """计算两个向量的余弦相似度."""
    if vec_a.numel() == 0 or vec_b.numel() == 0:
        return float("nan")
    if vec_a.shape != vec_b.shape:
        raise ValueError("Vectors must have the same shape")
    
    dot_product = torch.dot(vec_a, vec_b).item()
    norm_a = vec_a.norm(p=2).item()
    norm_b = vec_b.norm(p=2).item()
    
    denominator = norm_a * norm_b
    if denominator < eps:
        return float("nan")
    
    return dot_product / denominator


def _get_all_layer_indices(state_dict: Dict[str, torch.Tensor]) -> List[int]:
    """从state_dict中提取所有层索引."""
    layer_indices = set()
    for name in state_dict.keys():
        layer_idx = _extract_layer_index(name)
        if layer_idx is not None:
            layer_indices.add(layer_idx)
    return sorted(layer_indices)


def _load_state_dict(model_path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """加载模型并返回state_dict."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device if device == "cpu" else None,
    )
    if device != "cpu":
        model = model.to(device)
    state = {key: value.detach().cpu().float() for key, value in model.state_dict().items()}
    del model
    if device != "cpu":
        torch.cuda.empty_cache()
    return state


def _save_results(rows: List[dict], output_dir: Path) -> None:
    """保存结果到CSV和JSON文件."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "weight_cosine_similarity.csv"
    json_path = output_dir / "weight_cosine_similarity.json"
    
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def _plot_results(rows: List[dict], output_dir: Path) -> None:
    """绘制余弦相似度结果."""
    if not rows:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    
    layers = [row["layer_index"] for row in rows]
    cosine_sims = [row["cosine_similarity"] for row in rows]
    delta1_norms = [row["delta1_norm"] for row in rows]
    delta2_norms = [row["delta2_norm"] for row in rows]
    
    plt.style.use("default")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # 余弦相似度图
    axes[0].plot(layers, cosine_sims, marker="o", color="#4C72B0", linewidth=2, markersize=6)
    axes[0].axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[0].set_title("Weight Change Cosine Similarity: (defense1 - base) vs (defense2 - defense1)", fontsize=12)
    axes[0].set_xlabel("Layer Index", fontsize=10)
    axes[0].set_ylabel("Cosine Similarity", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].set_ylim(-1.1, 1.1)
    
    # 权重变化范数图
    axes[1].plot(layers, delta1_norms, marker="o", label="||defense1 - base||", color="#55A868", linewidth=2, markersize=6)
    axes[1].plot(layers, delta2_norms, marker="s", label="||defense2 - defense1||", color="#C44E52", linewidth=2, markersize=6)
    axes[1].set_title("Weight Change Norms by Layer", fontsize=12)
    axes[1].set_xlabel("Layer Index", fontsize=10)
    axes[1].set_ylabel("L2 Norm", fontsize=10)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].set_yscale("log")
    
    fig.tight_layout()
    fig.savefig(output_dir / "weight_cosine_similarity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/finetune/Meta-Llama-3-8B-Instruct-tofu",
        help="Base model path",
    )
    parser.add_argument(
        "--defense1",
        default="/data/xiangtao/projects/crossdefense/code/defense/fairness/BiasUnlearn/weights/Meta-Llama-3-8B-Instruct-tofu-Unbias",
        help="First defense model path",
    )
    parser.add_argument(
        "--defense2",
        default="/data/xiangtao/projects/crossdefense/code/defense/safety/Continuous-AdvTrain/weights/Meta-Llama-3-8B-Instruct-tofu-Unbias-CAT",
        help="Second defense model path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for analysis outputs",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="Meta-Llama-3-8B-Instruct-tofu-Unbias-CAT",
        help="Run identifier used to create a subdirectory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for model loading",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable plot generation",
    )
    return parser.parse_args()


def compute_weight_change_cosine_similarity(args: argparse.Namespace) -> None:
    """计算权重变化的余弦相似度."""
    print("Loading models...")
    base_state = _load_state_dict(args.base, device=args.device)
    defense1_state = _load_state_dict(args.defense1, device=args.device)
    defense2_state = _load_state_dict(args.defense2, device=args.device)
    
    # 获取所有层索引
    base_layers = set(_get_all_layer_indices(base_state))
    defense1_layers = set(_get_all_layer_indices(defense1_state))
    defense2_layers = set(_get_all_layer_indices(defense2_state))
    
    common_layers = sorted(base_layers & defense1_layers & defense2_layers)
    if not common_layers:
        raise ValueError("No common layers found across all three models")
    
    print(f"Found {len(common_layers)} common layers: {common_layers[0]} to {common_layers[-1]}")
    
    rows: List[dict] = []
    for layer_idx in common_layers:
        print(f"Processing layer {layer_idx}...")
        
        # 计算 delta1 = defense1 - base
        delta1 = _compute_layer_weight_delta(base_state, defense1_state, layer_idx)
        
        # 计算 delta2 = defense2 - defense1
        delta2 = _compute_layer_weight_delta(defense1_state, defense2_state, layer_idx)
        
        # 计算余弦相似度
        cosine_sim = _compute_cosine_similarity(delta1, delta2)
        
        # 计算范数
        delta1_norm = delta1.norm(p=2).item() if delta1.numel() > 0 else 0.0
        delta2_norm = delta2.norm(p=2).item() if delta2.numel() > 0 else 0.0
        
        rows.append({
            "layer_index": layer_idx,
            "layer_name": f"model.layers.{layer_idx}",
            "cosine_similarity": cosine_sim,
            "delta1_norm": delta1_norm,
            "delta2_norm": delta2_norm,
            "delta1_numel": delta1.numel(),
            "delta2_numel": delta2.numel(),
        })
    
    output_dir = args.output / args.run_name
    _save_results(rows, output_dir)
    print(f"Results saved to {output_dir}")
    
    if not args.no_plot:
        _plot_results(rows, output_dir)
        print(f"Plot saved to {output_dir / 'weight_cosine_similarity.png'}")
    
    # 打印摘要
    print("\n=== Summary ===")
    valid_cosines = [r["cosine_similarity"] for r in rows if not (isinstance(r["cosine_similarity"], float) and (r["cosine_similarity"] != r["cosine_similarity"]))]
    if valid_cosines:
        print(f"Mean cosine similarity: {sum(valid_cosines) / len(valid_cosines):.4f}")
        print(f"Min cosine similarity: {min(valid_cosines):.4f}")
        print(f"Max cosine similarity: {max(valid_cosines):.4f}")


def main() -> None:
    args = parse_args()
    compute_weight_change_cosine_similarity(args)


if __name__ == "__main__":
    main()

