"""
Quantify Sequential Defense Interaction (Synergy/Suppression) via Parameter Projection.
Supports batch processing for multiple defense pairs.
Granularity: Layer-wise (Aggregated).
Metric: Interaction Coefficient (rho) and Scalar Projection.
"""

from __future__ import annotations

import argparse
import json
import gc
import re
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

# Ensure we can import the provided analysis module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.parameter_deltas import load_state_dict 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Mode 1: Single Pair (Legacy support)
    parser.add_argument("--base", default=None, help="Path to Base LLM")
    parser.add_argument("--defense1", default=None, help="Path to Defense 1")
    parser.add_argument("--defense2", default=None, help="Path to Defense 2")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path for single run")
    
    # Mode 2: Batch Processing (Recommended)
    # Default config path updated to your specific path
    parser.add_argument("--config", type=Path, default="/data/xiangtao/projects/crossdefense/code/mechanistic_analysis/src/configs/privacy_defense_pairs.json", 
                        help="Path to JSON config file containing list of experiments.")
    parser.add_argument("--output_dir", type=Path, default="/data/xiangtao/projects/crossdefense/code/mechanistic_analysis/results/10-defense_inter/privacy", help="Directory to save batch results")
    
    parser.add_argument("--device", default="cpu", help="Device for loading (recommend cpu to save VRAM)")
    return parser.parse_args()

def get_layer_index(name: str) -> int:
    """Robust regex to extract layer index from various architectures."""
    match = re.search(r"\.(layers|h|blocks)\.(\d+)\.", name)
    if match:
        return int(match.group(2))
    return -1

def compute_metrics(u1_flat: torch.Tensor, u2_flat: torch.Tensor, eps=1e-12) -> Dict[str, float]:
    """Compute geometric interaction metrics between two update vectors."""
    dot = torch.dot(u1_flat, u2_flat).item()
    norm1_sq = torch.norm(u1_flat).pow(2).item()
    norm2_sq = torch.norm(u2_flat).pow(2).item()
    
    # Return raw stats for aggregation
    return {
        "dot": dot,
        "norm1_sq": norm1_sq,
        "norm2_sq": norm2_sq
    }

def process_triplet_safe(base_path: str, def1_path: str, def2_path: str, device: str) -> List[Dict[str, Any]]:
    """
    Memory-safe implementation of triplet processing with Layer-Level Aggregation.
    """
    # 1. Load Base and Def1
    print(f"  Loading Base: {base_path}")
    base_sd = load_state_dict(base_path, device=device)
    print(f"  Loading Defense 1: {def1_path}")
    def1_sd = load_state_dict(def1_path, device=device)
    
    # 2. Compute u1 and store sparse
    u1_storage = {}
    common_keys = set(base_sd.keys()) & set(def1_sd.keys())
    
    for key in common_keys:
        # Filter: Weights only, no Norm/Bias/Embeddings
        # Strictly focusing on transformation layers
        if "weight" not in key or any(x in key for x in ["norm", "ln_", "bias", "embed", "wte", "wpe"]):
            continue
        u1_storage[key] = def1_sd[key].float() - base_sd[key].float()
        
    # 3. Free Base
    del base_sd
    gc.collect()
    
    # 4. Load Def2
    print(f"  Loading Defense 2: {def2_path}")
    def2_sd = load_state_dict(def2_path, device=device)
    
    # 5. Compute metrics layer by layer
    # Structure: layer_idx -> {dot, norm1_sq, norm2_sq}
    layer_stats = {}
    
    keys_final = sorted(list(set(u1_storage.keys()) & set(def1_sd.keys()) & set(def2_sd.keys())))
    
    for key in tqdm(keys_final, desc="Computing Metrics"):
        # u2 = D2 - D1
        u2 = def2_sd[key].float() - def1_sd[key].float()
        u1 = u1_storage[key]
        
        # Flatten
        u1_flat = u1.view(-1)
        u2_flat = u2.view(-1)
        
        # Compute Dot and Norms
        metrics = compute_metrics(u1_flat, u2_flat)
        
        # Aggregate by Layer (No module distinction)
        layer_idx = get_layer_index(key)
        
        if layer_idx == -1: continue
        
        if layer_idx not in layer_stats:
            layer_stats[layer_idx] = {"dot": 0.0, "norm1_sq": 0.0, "norm2_sq": 0.0}
            
        # Accumulate the raw values
        # Mathematically: dot(A+B, C+D) = dot(A,C) + dot(B,D) if A,B and C,D are disjoint parts of a larger vector
        layer_stats[layer_idx]["dot"] += metrics["dot"]
        layer_stats[layer_idx]["norm1_sq"] += metrics["norm1_sq"]
        layer_stats[layer_idx]["norm2_sq"] += metrics["norm2_sq"]

    # 6. Cleanup
    del def1_sd, def2_sd, u1_storage
    gc.collect()
    
    # 7. Finalize Results
    results = []
    for layer_idx in sorted(layer_stats.keys()):
        stats = layer_stats[layer_idx]
        
        norm1 = np.sqrt(stats["norm1_sq"])
        norm2 = np.sqrt(stats["norm2_sq"])
        dot = stats["dot"]
        
        # Interaction Coefficient (Cosine Similarity of the WHOLE layer)
        rho = dot / (norm1 * norm2 + 1e-12)
        
        # Scalar Projection Magnitude
        scalar_proj = dot / (norm1 + 1e-12)
        
        results.append({
            "layer": layer_idx,
            "rho": rho,
            "scalar_proj": scalar_proj,
            "norm_u1": norm1,
            "norm_u2": norm2
        })
        
    return results

def main():
    args = parse_args()
    
    experiments = []
    
    # Handle Config Mode
    if args.config:
        with open(args.config, 'r') as f:
            experiments = json.load(f)
        if not args.output_dir:
            raise ValueError("Must specify --output_dir when using --config")
        args.output_dir.mkdir(parents=True, exist_ok=True)
    # Handle Single Mode
    elif args.base and args.defense1 and args.defense2 and args.output:
        experiments = [{
            "id": "single_run",
            "base": args.base,
            "defense1": args.defense1,
            "defense2": args.defense2,
            "output_file": args.output
        }]
    else:
        print("Error: Must provide either --config/--output_dir OR --base/--defense1/--defense2/--output")
        return

    print(f"Found {len(experiments)} experiments to process.")
    
    for i, exp in enumerate(experiments):
        exp_id = exp.get("id", f"exp_{i}")
        print(f"\n=== Processing Experiment {i+1}/{len(experiments)}: {exp_id} ===")
        
        base_p = exp["base"]
        d1_p = exp["defense1"]
        d2_p = exp["defense2"]
        
        # Determine output path
        if "output_file" in exp:
            out_path = Path(exp["output_file"])
        else:
            out_path = args.output_dir / f"{exp_id}_projection.json"
            
        if out_path.exists():
            print(f"Skipping {exp_id}, output exists at {out_path}")
            continue
            
        try:
            results = process_triplet_safe(base_p, d1_p, d2_p, args.device)
            
            # Save results
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"experiment_id": exp_id, "per_layer": results}, f, indent=2)
            print(f"Saved results to {out_path}")
            
        except Exception as e:
            print(f"Failed to process experiment {exp_id}: {e}")
            import traceback
            traceback.print_exc()
            
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()