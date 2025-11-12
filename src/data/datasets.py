"""Utilities for loading and preparing datasets used in the defense analysis pipeline.

The dataset utilities focus on building the collections required by the research plan:
1. Malicious request dataset (D_mal).
2. Normal request dataset (D_norm).
3. Privacy-sensitive dataset (D_priv).

All datasets are stored as JSONL files with at minimum a ``text`` field.  The helper
functions expose simple PyTorch ``Dataset`` wrappers so they can be combined with the
rest of the analysis code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import json

import torch
from torch.utils.data import Dataset


@dataclass
class JsonlRecord:
    """Simple container for a JSONL record.

    Attributes
    ----------
    text:
        The raw prompt string stored in the dataset.
    metadata:
        Optional metadata dictionary.  All keys except ``text`` are stored here.
    """

    text: str
    metadata: Optional[dict] = None


def read_jsonl(path: Path | str) -> Iterator[JsonlRecord]:
    """Stream JSONL records from ``path``.

    Parameters
    ----------
    path:
        File system path to a JSONL file.  Each line must be a JSON object containing at
        least a ``text`` field.

    Yields
    ------
    JsonlRecord
        Parsed record containing the raw text and any additional metadata.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            text = payload.pop("text")
            metadata = payload if payload else None
            yield JsonlRecord(text=text, metadata=metadata)


class RequestDataset(Dataset):
    """PyTorch dataset that returns tokenized prompts.

    The dataset handles tokenization lazily to support large corpora.  Tokenization is
    performed on demand using the provided tokenizer.
    """

    def __init__(
        self,
        records: Sequence[JsonlRecord],
        tokenizer,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        encoded = self.tokenizer(
            record.text,
            max_length=self.max_length,
            padding="max_length" if self.padding else False,
            truncation=self.truncation,
            return_tensors="pt",
        )
        batch = {key: value.squeeze(0) for key, value in encoded.items()}
        if record.metadata is not None:
            batch["metadata"] = record.metadata
        return batch


def load_dataset(
    path: Path | str,
    tokenizer,
    max_length: int = 512,
) -> RequestDataset:
    """Load a dataset from ``path`` and return a :class:`RequestDataset` instance."""

    records = list(read_jsonl(path))
    return RequestDataset(records, tokenizer=tokenizer, max_length=max_length)


@dataclass
class DualDataset:
    """Container holding the paired datasets required by the analysis pipeline."""

    normal: RequestDataset
    malicious: RequestDataset
    privacy: Optional[RequestDataset] = None


def build_dual_dataset(
    normal_path: Path | str,
    malicious_path: Path | str,
    tokenizer,
    privacy_path: Optional[Path | str] = None,
    max_length: int = 512,
) -> DualDataset:
    """Load the datasets needed for safety/privacy concept extraction.

    Parameters
    ----------
    normal_path:
        Path to the neutral prompts (``D_norm``).
    malicious_path:
        Path to the malicious prompts (``D_mal``).
    tokenizer:
        Tokenizer compatible with the target language model.
    privacy_path:
        Optional path to the privacy-sensitive dataset (``D_priv``).
    max_length:
        Maximum token length for the tokenizer.
    """

    normal = load_dataset(normal_path, tokenizer=tokenizer, max_length=max_length)
    malicious = load_dataset(malicious_path, tokenizer=tokenizer, max_length=max_length)
    privacy = None
    if privacy_path is not None:
        privacy = load_dataset(privacy_path, tokenizer=tokenizer, max_length=max_length)
    return DualDataset(normal=normal, malicious=malicious, privacy=privacy)
