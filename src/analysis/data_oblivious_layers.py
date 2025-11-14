"""Data-oblivious critical layer detection utilities.

This module adapts the open-source implementation that accompanies the paper
"Data-Oblivious Critical Layers" (https://arxiv.org/abs/2506.00382) for use in
our analysis toolkit.  Unlike the cosine-based procedure implemented in
``key_layers.py``, the routines here do not require access to a curated dataset
of normal/risky prompts.  Instead we probe a language model with synthetic input
batches sampled directly from its vocabulary and measure how strongly each
transformer block responds to random supervision signals.  Layers that react the
most to these input-agnostic probes are marked as critical.

The public reference implementation targets a bespoke training pipeline; this
file reworks the core ideas into lightweight, framework-agnostic utilities that
operate on a Hugging Face ``PreTrainedModel``.  No network calls are performed at
runtime so the code can be used in air-gapped environments.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass
class LayerProbeStatistics:
    """Aggregated activation/gradient magnitudes for a single transformer block."""

    activation_norm: float
    gradient_norm: float
    saliency: float
    steps: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "activation_norm": self.activation_norm,
            "gradient_norm": self.gradient_norm,
            "saliency": self.saliency,
            "steps": float(self.steps),
        }


@dataclass
class DataObliviousCriticalLayerReport:
    """Container returned by :func:`identify_data_oblivious_critical_layers`."""

    layer_names: List[str]
    scores: List[float]
    normalized_scores: List[float]
    ranked_layers: List[int]
    statistics: Dict[str, LayerProbeStatistics]

    def topk(self, k: int) -> List[Tuple[str, float]]:
        """Return the top ``k`` layers together with their normalised scores."""

        if k <= 0:
            return []
        k = min(k, len(self.layer_names))
        return [
            (self.layer_names[idx], self.normalized_scores[idx])
            for idx in self.ranked_layers[:k]
        ]

    def to_dict(self) -> Dict[str, object]:
        return {
            "layer_names": list(self.layer_names),
            "scores": list(self.scores),
            "normalized_scores": list(self.normalized_scores),
            "ranked_layers": list(self.ranked_layers),
            "statistics": {
                name: stats.to_dict() for name, stats in self.statistics.items()
            },
        }


def _default_transformer_layers(model: torch.nn.Module) -> Tuple[List[str], List[torch.nn.Module]]:
    """Return canonical transformer block modules for common HF architectures."""

    # LLaMA/OPT style (decoder only)
    with contextlib.suppress(AttributeError):
        layers = getattr(getattr(model, "model"), "layers")
        return [f"model.layers.{idx}" for idx in range(len(layers))], list(layers)

    # GPT2 style
    with contextlib.suppress(AttributeError):
        layers = getattr(getattr(model, "transformer"), "h")
        return [f"transformer.h.{idx}" for idx in range(len(layers))], list(layers)

    # Encoder-decoder BERT/RoBERTa style
    with contextlib.suppress(AttributeError):
        layers = getattr(getattr(model, "encoder"), "layer")
        return [f"encoder.layer.{idx}" for idx in range(len(layers))], list(layers)

    raise ValueError(
        "Could not automatically determine transformer blocks. Please pass"
        " `layer_modules` explicitly."
    )


@dataclass
class _HookCache:
    activation: Optional[torch.Tensor] = None
    activation_norm_sum: float = 0.0
    gradient_norm_sum: float = 0.0
    saliency_sum: float = 0.0
    steps: int = 0


def _register_probes(
    layers: Sequence[torch.nn.Module],
) -> Tuple[List[torch.utils.hooks.RemovableHandle], List[_HookCache]]:
    caches: List[_HookCache] = [_HookCache() for _ in layers]
    handles: List[torch.utils.hooks.RemovableHandle] = []

    for idx, (layer, cache) in enumerate(zip(layers, caches)):
        def forward_hook(module, inputs, output, *, _cache=cache):
            if hasattr(output, 'last_hidden_state'):
                tensor = output.last_hidden_state
            else:
                tensor = output[0] if isinstance(output, (tuple, list)) else output
            _cache.activation = tensor.detach()

        def backward_hook(module, grad_input, grad_output, *, _cache=cache):
            if hasattr(grad_output, 'last_hidden_state'):
                grad_tensor = grad_output.last_hidden_state
            else:
                grad_tensor = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
            if grad_tensor is None:
                return
            grad_tensor = grad_tensor.detach()
            if _cache.activation is None:
                return
            activation = _cache.activation
            _cache.activation = None
            # Compute norms averaged across batch/sequence dimensions.
            _cache.activation_norm_sum += activation.pow(2).mean().item()
            _cache.gradient_norm_sum += grad_tensor.pow(2).mean().item()
            # Accumulate element-wise saliency averaged across tokens and batch.
            saliency = (activation.abs() * grad_tensor.abs()).sum(dim=-1).mean().item()
            _cache.saliency_sum += saliency
            _cache.steps += 1

        handles.append(layer.register_forward_hook(forward_hook))
        handles.append(layer.register_full_backward_hook(backward_hook))

    return handles, caches


def _generate_random_batch(
    *,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    if generator is not None:
        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device, generator=generator
        )
    else:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def identify_data_oblivious_critical_layers(
    model: torch.nn.Module,
    *,
    layer_modules: Optional[Sequence[torch.nn.Module]] = None,
    layer_names: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 4,
    seq_len: Optional[int] = None,
    num_batches: int = 32,
    vocab_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> DataObliviousCriticalLayerReport:
    """Score layers by gradient saliency under random supervision.

    Parameters
    ----------
    model:
        The Hugging Face model to inspect.  ``output_hidden_states`` must be
        supported.  The model is placed in ``eval`` mode and gradients are
        computed with respect to the random loss constructed below.
    layer_modules / layer_names:
        Optional explicit specification of modules to probe.  When omitted the
        function attempts to discover transformer blocks automatically using
        ``_default_transformer_layers``.
    device:
        Target device.  When ``None`` the current ``model.device`` is used.
    batch_size, seq_len, num_batches:
        Control how many synthetic prompts are used to probe the network.
    vocab_size:
        Overrides the vocabulary size used to sample random tokens.  Defaults to
        ``model.config.vocab_size``.
    seed:
        Optional random seed for reproducibility.
    """

    model.eval()
    if device is None:
        device = next(model.parameters()).device
    if layer_modules is None or layer_names is None:
        auto_names, auto_layers = _default_transformer_layers(model)
        if layer_modules is None:
            layer_modules = auto_layers
        if layer_names is None:
            layer_names = auto_names
    if len(layer_modules) != len(layer_names):
        raise ValueError("layer_modules and layer_names must have the same length")
    if seq_len is None:
        seq_len = getattr(getattr(model, "config", object()), "max_position_embeddings", 128)
    if vocab_size is None:
        vocab_size = getattr(getattr(model, "config", object()), "vocab_size", None)
        if vocab_size is None:
            raise ValueError("Could not infer vocab size; please pass it explicitly.")

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    handles, caches = _register_probes(layer_modules)
    statistics: Dict[str, LayerProbeStatistics] = {}

    try:
        for _ in range(num_batches):
            batch = _generate_random_batch(
                vocab_size=vocab_size,
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                generator=generator,
            )

            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, output_hidden_states=True, use_cache=False)
            logits = outputs.logits
            target = torch.randint(
                logits.size(-1),
                (logits.size(0),),
                device=device,
                generator=generator,
            )
            loss = F.cross_entropy(logits[:, -1, :], target)
            model.zero_grad(set_to_none=True)
            loss.backward()

        scores: List[float] = []
        for name, cache in zip(layer_names, caches):
            steps = max(cache.steps, 1)
            activation_norm = cache.activation_norm_sum / steps
            gradient_norm = cache.gradient_norm_sum / steps
            saliency = cache.saliency_sum / steps
            statistics[name] = LayerProbeStatistics(
                activation_norm=activation_norm,
                gradient_norm=gradient_norm,
                saliency=saliency,
                steps=steps,
            )
            scores.append(saliency)

        total = sum(scores) or 1.0
        normalized_scores = [score / total for score in scores]
        ranked_layers = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    finally:
        for handle in handles:
            handle.remove()

    return DataObliviousCriticalLayerReport(
        layer_names=list(layer_names),
        scores=scores,
        normalized_scores=normalized_scores,
        ranked_layers=ranked_layers,
        statistics=statistics,
    )


__all__ = [
    "LayerProbeStatistics",
    "DataObliviousCriticalLayerReport",
    "identify_data_oblivious_critical_layers",
]
