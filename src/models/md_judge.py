"""Wrapper around the MD-Judge evaluator for safety scoring."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from transformers import AutoTokenizer

try:  # pragma: no cover - optional dependency
    from vllm import LLM, SamplingParams
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "The MD-Judge evaluator requires the `vllm` package. "
        "Install vLLM to enable automatic safety scoring."
    ) from exc


_MD_JUDGE_MODEL = os.getenv("MD_JUDGE_MODEL", "OpenSafetyLab/MD-Judge-v0_2-internlm2_7b")
_TP_SIZE = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
_GPU_UTIL = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.4"))


def _extract_score(text: str) -> int:
    payload = text.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```.*?\n|\n```$", "", payload, flags=re.S)
    match = re.search(r"#thescore:\s*([1-5])\b", payload, flags=re.I)
    if match:
        return int(match.group(1))
    fallback = re.search(r"\b([1-5])\b", payload)
    if fallback:
        return int(fallback.group(1))
    return 2


@dataclass
class MDJudgeResult:
    scores: List[int]


class MDJudgeEvaluator:
    """Evaluate (prompt, response) pairs with the MD-Judge model."""

    def __init__(
        self,
        model_path: str = _MD_JUDGE_MODEL,
        tensor_parallel_size: int = _TP_SIZE,
        gpu_memory_utilization: float = _GPU_UTIL,
    ) -> None:
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def _build_prompt(self, instruction: str, response: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def score(self, pairs: Sequence[Tuple[str, str]], max_tokens: int = 256) -> MDJudgeResult:
        prompts = [self._build_prompt(q, a) for q, a in pairs]
        scores: List[int] = []
        params = SamplingParams(max_tokens=max_tokens)
        for prompt in prompts:
            outputs = self.llm.generate(prompt, sampling_params=params)
            text = outputs[0].outputs[0].text.strip() if outputs[0].outputs else ""
            scores.append(_extract_score(text))
        return MDJudgeResult(scores=scores)
