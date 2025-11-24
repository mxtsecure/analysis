"""Step 2: quantify overlap between safety and privacy concept vectors."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import json
from typing import Dict
import sys
import os
# 导入路径，确保 ConceptVector 可用
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 假设 analysis.concept_vectors 中的 ConceptVector 和 cosine_similarity 已经定义
from analysis.concept_vectors import ConceptVector, cosine_similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concepts_vector_path", type=Path, default=Path("/data/xiangtao/projects/crossdefense/code/analysis_results/03-concepts_vector/gemma-2-2b-it-tofu/accurate"), help="Path to concepts vector directory")
    parser.add_argument("--concepts_vector_overlap_path", type=Path, default=Path("/data/xiangtao/projects/crossdefense/code/analysis_results/04-concepts_vector_overlap/gemma-2-2b-it-tofu/accurate/result_pca.json"), help="Path to concepts vector directory")
    return parser.parse_args()


def load_concept(path: Path) -> ConceptVector:
    # 确保 ConceptVector 类在 Step 2 脚本的执行环境中正确导入
    return torch.load(path, map_location="cpu", weights_only=False)


def main() -> None:  # pragma: no cover - CLI
    args = parse_args()
    results: Dict[str, Dict[str, float]] = {}
    for layer in os.listdir(args.concepts_vector_path):
        if not layer.startswith("model_layers_"):
            continue
        print(f"Layer: {layer}")
        v_safety = load_concept(args.concepts_vector_path / layer / "v_safety.pt")
        v_privacy = load_concept(args.concepts_vector_path / layer / "v_privacy.pt")

        # --- [修改 1] 从 raw_norm 属性中获取原始范数 ---
        # 如果 raw_norm 不存在 (例如，旧文件)，则使用 L2 范数 (此时通常为 1.0)
        safety_raw_norm = v_safety.raw_norm if v_safety.raw_norm is not None else v_safety.direction.norm(p=2).item()
        privacy_raw_norm = v_privacy.raw_norm if v_privacy.raw_norm is not None else v_privacy.direction.norm(p=2).item()
        
        # 概念向量 direction 已经是归一化的
        # 注意：这里的 safety_norm 和 privacy_norm 应该总是接近 1.0
        safety_norm_unit = v_safety.direction.norm(p=2).item() 
        privacy_norm_unit = v_privacy.direction.norm(p=2).item() 

        # 计算重叠，使用归一化的方向进行点积
        raw_overlap_unit = torch.dot(v_safety.direction, v_privacy.direction).item()
        similarity = cosine_similarity(v_safety, v_privacy)

        # magnitude_weighted_overlap 仍然使用归一化的范数 (max(1.0, 1.0))，它会等于 similarity
        # 除非您想使用 raw_norm 来计算，但通常这个指标是基于单位向量的
        magnitude_weighted_overlap = raw_overlap_unit / (max(safety_norm_unit, privacy_norm_unit) + 1e-12)

        # --- [修改 2] 计算投影大小 (有符号长度) ---
        # 投影大小 L = (a · b_unit) / |b_unit| = (a · b_unit) / 1
        # 但我们使用 raw_overlap_unit (v_safety_unit · v_privacy_unit)，它等于 cos(theta)
        
        # Safety 在 Privacy 上的投影大小 (L_S->P)
        # L_S->P = |v_safety| * cos(theta)
        projection_safety_on_privacy_magnitude = safety_raw_norm * similarity
        
        # Privacy 在 Safety 上的投影大小 (L_P->S)
        # L_P->S = |v_privacy| * cos(theta)
        projection_privacy_on_safety_magnitude = privacy_raw_norm * similarity


        # --- [修改 3] 计算相比于自身概念向量变化幅度的百分比 ---
        # Safety 投影百分比 = (|v_safety| * cos(theta)) / |v_safety| * 100% = cos(theta) * 100%
        # 这就是余弦相似度，但为了满足用户要求，我们基于投影大小和原始范数进行计算
        
        if safety_raw_norm > 1e-12:
            safety_projection_on_privacy_percent = (projection_safety_on_privacy_magnitude / safety_raw_norm) * 100
        else:
            safety_projection_on_privacy_percent = 0.0

        if privacy_raw_norm > 1e-12:
            privacy_projection_on_safety_percent = (projection_privacy_on_safety_magnitude / privacy_raw_norm) * 100
        else:
            privacy_projection_on_safety_percent = 0.0


        results[layer] = {
            "cosine_similarity": round(similarity, 6),
            "safety_raw_norm": round(safety_raw_norm, 6),
            "privacy_raw_norm": round(privacy_raw_norm, 6),
            "magnitude_weighted_overlap": round(magnitude_weighted_overlap, 6),
            
            "projection_safety_on_privacy_magnitude": round(projection_safety_on_privacy_magnitude, 6),
            "projection_privacy_on_safety_magnitude": round(projection_privacy_on_safety_magnitude, 6),
            
            "safety_projection_on_privacy_percent": round(safety_projection_on_privacy_percent, 4),
            "privacy_projection_on_safety_percent": round(privacy_projection_on_safety_percent, 4),
        }

        print(
            "  "
            f"safety_norm={safety_raw_norm:.6f}, privacy_norm={privacy_raw_norm:.6f}, "
            f"cosine_similarity={similarity:.6f}, "
            f"weighted_overlap={magnitude_weighted_overlap:.6f}"
        )
        print(
            "  "
            f"Safety → Privacy: magnitude={projection_safety_on_privacy_magnitude:.6f}, "
            f"relative_change={safety_projection_on_privacy_percent:.4f}%"
        )
        print(
            "  "
            f"Privacy → Safety: magnitude={projection_privacy_on_safety_magnitude:.6f}, "
            f"relative_change={privacy_projection_on_safety_percent:.4f}%"
        )
        
    args.concepts_vector_overlap_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.concepts_vector_overlap_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":  # pragma: no cover
    main()