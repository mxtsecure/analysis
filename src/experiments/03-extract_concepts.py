"""Script for Step 1: extracting purified safety/privacy concept vectors."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.concept_vectors import (
    ConceptVector,
    aggregate_difference,
    purify_concept_vector,
)
from data.datasets import build_dual_dataset
from models.activations import compute_activation_difference, forward_dataset


def _normalize_layer_identifier(raw: str) -> str:
    """Translate a CLI ``--layer`` value into a module path understood by hooks."""

    stripped = raw.strip()
    if not stripped:
        raise ValueError("Layer names must be non-empty")
    if stripped.isdigit():
        # The majority of our experiments target transformer blocks addressed by
        # their numeric index. ``model.layers`` is the common location for these
        # blocks across the supported Hugging Face causal LM architectures, so we
        # expose the ergonomic shorthand ``--layer 12`` which maps to
        # ``model.layers.12``.
        return f"model.layers.{stripped}"
    return stripped


def _parse_layers(layer_argument: Sequence[str] | str) -> list[str]:
    """Normalize the ``--layer`` argument into a list of module names."""

    if isinstance(layer_argument, str):
        raw_entries = [layer_argument]
    else:
        raw_entries = list(layer_argument)
    layers: list[str] = []
    for entry in raw_entries:
        for part in entry.split(","):
            name = part.strip()
            if name:
                layers.append(_normalize_layer_identifier(name))
    if not layers:
        raise ValueError("At least one layer name must be provided via --layer")
    print(f"Capturing activations from layers: {layers}")
    return layers


def _layer_output_dir(base: Path, layer: str, multiple_layers: bool) -> Path:
    """Return the directory where vectors for ``layer`` should be stored."""

    if not multiple_layers:
        return base
    safe_layer = layer.replace("/", "_").replace(".", "_")
    return base / safe_layer


def _as_layer_mapping(value, layers: Sequence[str]):
    """Ensure activation outputs are returned as a mapping keyed by layer."""

    if isinstance(value, dict):
        return value
    if len(layers) != 1:
        raise ValueError("Expected multiple layers to yield a mapping of activations")
    return {layers[0]: value}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/finetune/Llama-3.2-1B-Instruct-tofu", help="Base model name or path")
    parser.add_argument("--safety", default="/data/xiangtao/projects/crossdefense/code/defense/safety/DPO/DPO_models/different/Llama-3.2-1B-Instruct-tofu-DPO", help="Safety fine-tuned model")
    parser.add_argument("--privacy", default="/data/xiangtao/projects/crossdefense/code/defense/privacy/open-unlearning/saves/unlearn/Llama-3.2-1B-Instruct-tofu/Llama-3.2-1B-Instruct-tofu-NPO", help="Privacy fine-tuned model")
    parser.add_argument("--normal", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/normal.jsonl", help="Path to D_norm JSONL")
    parser.add_argument("--malicious", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/safety.jsonl", help="Path to D_mal JSONL")
    parser.add_argument("--privacy-data", default="/data/xiangtao/projects/crossdefense/code/analysis/datasets/risk_data/privacy.jsonl", help="Path to D_priv JSONL")
    parser.add_argument(
        "--layer",
        dest="layer",
        action="append",
        default="0,1,2,3,4,5,6,7,8,9",
        help=(
            "Module name(s) for activation capture. Repeat the flag or provide a "
            "comma-separated list to capture multiple layers."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("/data/xiangtao/projects/crossdefense/code/analysis/results/01-concepts_vector/Llama-3.2-1B-Instruct-tofu/accurate"), help="Output directory")
    parser.add_argument(
        "--method",
        choices=["mean", "pca"],
        default="mean",
        help="Aggregation method for concept vectors",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "accurate"],
        default="accurate",
        help=(
            "Extraction mode: 'fast' computes concept vectors directly from the "
            "risk datasets, while 'accurate' performs the multi-dataset "
            "purification procedure."
        ),
    )
    args = parser.parse_args()
    # Normalise argparse's append behaviour so the rest of the code can treat the
    # attribute as either a sequence of strings or a single string when provided
    # programmatically.
    if len(args.layer) == 1:
        args.layer = args.layer[0]
    return args


def make_dataloaders(tokenizer, args) -> tuple[DataLoader, DataLoader, DataLoader]:
    datasets = build_dual_dataset(
        normal_path=args.normal,
        malicious_path=args.malicious,
        tokenizer=tokenizer,
        privacy_path=args.privacy_data,
    )
    if datasets.privacy is None:
        raise ValueError("Privacy dataset is required for concept extraction")
    normal_loader = DataLoader(datasets.normal, batch_size=args.batch_size)
    malicious_loader = DataLoader(datasets.malicious, batch_size=args.batch_size)
    privacy_loader = DataLoader(datasets.privacy, batch_size=args.batch_size)
    return normal_loader, malicious_loader, privacy_loader


def extract_concept_vectors(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    base_model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16).to(device)
    safety_model = AutoModelForCausalLM.from_pretrained(args.safety, dtype=torch.float16).to(device)
    privacy_model = AutoModelForCausalLM.from_pretrained(args.privacy, dtype=torch.float16).to(device)

    normal_loader, malicious_loader, privacy_loader = make_dataloaders(tokenizer, args)

    layers = _parse_layers(args.layer)

    def capture(model, dataloader, desc):
        return _as_layer_mapping(
            forward_dataset(
                model,
                dataloader,
                layers,
                device,
                desc,
            ),
            layers,
        )

    multiple_layers = len(layers) > 1
    results: dict[str, dict[str, ConceptVector]] = {}

    if args.mode == "fast":
        base_mal = capture(base_model, malicious_loader, "Base risk dataset")
        safety_mal = capture(safety_model, malicious_loader, "Safety risk dataset")

        base_priv = capture(base_model, privacy_loader, "Base privacy dataset")
        privacy_priv = capture(privacy_model, privacy_loader, "Privacy dataset")

        for layer in layers:
            mal_diff = compute_activation_difference(base_mal[layer], safety_mal[layer])
            v_safety = aggregate_difference(mal_diff, method=args.method)

            priv_diff = compute_activation_difference(base_priv[layer], privacy_priv[layer])
            v_privacy = aggregate_difference(priv_diff, method=args.method)

            output_dir = _layer_output_dir(args.output, layer, multiple_layers)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(v_safety, output_dir / "v_safety.pt")
            torch.save(v_privacy, output_dir / "v_privacy.pt")

            results[layer] = {
                "v_safety": v_safety,
                "v_privacy": v_privacy,
            }
    else:
        base_norm = capture(base_model, normal_loader, "Base D_norm")
        safety_norm = capture(safety_model, normal_loader, "Safety D_norm")

        base_mal = capture(base_model, malicious_loader, "Base D_mal")
        safety_mal = capture(safety_model, malicious_loader, "Safety D_mal")

        base_priv = capture(base_model, privacy_loader, "Base D_priv")
        privacy_priv = capture(privacy_model, privacy_loader, "Privacy D_priv")

        privacy_norm = capture(privacy_model, normal_loader, "Privacy D_norm")
        base_priv_norm = capture(base_model, normal_loader, "Base D_norm (privacy)")

        for layer in layers:
            norm_diff = compute_activation_difference(base_norm[layer], safety_norm[layer])
            mal_diff = compute_activation_difference(base_mal[layer], safety_mal[layer])

            v_norm = aggregate_difference(norm_diff, method=args.method)
            v_tilde_safety = aggregate_difference(mal_diff, method=args.method)
            v_safety = purify_concept_vector(v_tilde_safety, v_norm)

            priv_diff = compute_activation_difference(base_priv[layer], privacy_priv[layer])
            priv_norm_diff = compute_activation_difference(
                base_priv_norm[layer], privacy_norm[layer]
            )

            v_privacy_mixed = aggregate_difference(priv_diff, method=args.method)
            v_privacy_norm = aggregate_difference(priv_norm_diff, method=args.method)
            v_privacy = purify_concept_vector(v_privacy_mixed, v_privacy_norm)

            output_dir = _layer_output_dir(args.output, layer, multiple_layers)
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(v_norm, output_dir / "v_norm.pt")
            torch.save(v_safety, output_dir / "v_safety.pt")
            torch.save(v_privacy, output_dir / "v_privacy.pt")

            results[layer] = {
                "v_norm": v_norm,
                "v_safety": v_safety,
                "v_privacy": v_privacy,
            }

    if multiple_layers:
        return results
    first_layer = layers[0]
    return results[first_layer]


def main() -> None:  # pragma: no cover - CLI entrypoint
    args = parse_args()
    extract_concept_vectors(args)


if __name__ == "__main__":  # pragma: no cover
    main()
