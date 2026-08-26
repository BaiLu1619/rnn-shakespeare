"""Download and stream Tiny Shakespeare in character-level batches."""

from __future__ import annotations

import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch

DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DATASET_FILENAME = "input.txt"


def _validate_text(text: str) -> None:
    if len(text) < 1_000_000:
        raise RuntimeError("Tiny Shakespeare file is unexpectedly short")
    if len(set(text)) < 50:
        raise RuntimeError("Tiny Shakespeare vocabulary is unexpectedly small")
    if "First Citizen:" not in text:
        raise RuntimeError("downloaded file does not look like Tiny Shakespeare")


def download_tiny_shakespeare(data_dir: str | Path = "data/tinyshakespeare") -> Path:
    """Download Tiny Shakespeare and validate the resulting UTF-8 text."""
    destination_dir = Path(data_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / DATASET_FILENAME

    if not destination.is_file():
        partial = destination.with_suffix(".txt.part")
        print(f"downloading Tiny Shakespeare to: {destination.resolve()}")
        try:
            with urllib.request.urlopen(DATASET_URL, timeout=60) as response:
                with partial.open("wb") as output:
                    shutil.copyfileobj(response, output)
            partial.replace(destination)
        except Exception as error:
            partial.unlink(missing_ok=True)
            proxy = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
            proxy_hint = (
                f" Current HTTPS proxy: {proxy!r}."
                if proxy
                else " Check network access and retry."
            )
            raise RuntimeError(
                f"unable to download {DATASET_URL}.{proxy_hint}"
            ) from error

    try:
        text = destination.read_text(encoding="utf-8")
        _validate_text(text)
    except Exception as error:
        raise RuntimeError(f"invalid Tiny Shakespeare file: {destination}") from error

    print(f"Tiny Shakespeare is ready ({len(text):,} characters).")
    return destination


class CharacterVocabulary:
    """Bidirectional mapping between characters and integer token IDs."""

    def __init__(self, characters: list[str]) -> None:
        if not characters or len(set(characters)) != len(characters):
            raise ValueError("characters must be unique and non-empty")
        self.itos = list(characters)
        self.stoi = {character: index for index, character in enumerate(self.itos)}

    @classmethod
    def from_text(cls, text: str) -> "CharacterVocabulary":
        return cls(sorted(set(text)))

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> torch.Tensor:
        try:
            values = [self.stoi[character] for character in text]
        except KeyError as error:
            raise ValueError(
                f"character {error.args[0]!r} is not in the vocabulary"
            ) from error
        return torch.tensor(values, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor | list[int]) -> str:
        values = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        try:
            return "".join(self.itos[int(index)] for index in values)
        except (IndexError, TypeError) as error:
            raise ValueError("token ID is outside the vocabulary") from error


class CharacterBatchLoader:
    """Create continuous batch streams for truncated backpropagation through time."""

    def __init__(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> None:
        if tokens.ndim != 1 or tokens.dtype != torch.long:
            raise ValueError("tokens must be a one-dimensional torch.long tensor")
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("batch_size and sequence_length must be positive")

        block_size = batch_size * sequence_length
        self.num_batches = (len(tokens) - 1) // block_size
        if self.num_batches == 0:
            raise ValueError("split is too short for the batch and sequence lengths")
        usable = self.num_batches * block_size
        self.inputs = tokens[:usable].view(batch_size, -1)
        self.targets = tokens[1 : usable + 1].view(batch_size, -1)
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for index in range(self.num_batches):
            start = index * self.sequence_length
            stop = start + self.sequence_length
            yield self.inputs[:, start:stop], self.targets[:, start:stop]


@dataclass
class ShakespeareData:
    train: CharacterBatchLoader
    validation: CharacterBatchLoader
    test: CharacterBatchLoader
    vocabulary: CharacterVocabulary
    split_sizes: dict[str, int]


def get_shakespeare_data(
    data_dir: str | Path,
    *,
    sequence_length: int = 100,
    batch_size: int = 64,
    train_fraction: float = 0.90,
    validation_fraction: float = 0.05,
    download: bool = True,
) -> ShakespeareData:
    """Build chronological splits and continuous mini-batch streams."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must sum to less than 1")

    path = (
        download_tiny_shakespeare(data_dir)
        if download
        else Path(data_dir) / DATASET_FILENAME
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset not found at {path}; run 'python main.py --download-data' first"
        )
    text = path.read_text(encoding="utf-8")
    _validate_text(text)

    vocabulary = CharacterVocabulary.from_text(text)
    tokens = vocabulary.encode(text)
    train_end = int(len(tokens) * train_fraction)
    validation_end = train_end + int(len(tokens) * validation_fraction)
    split_tokens = {
        "train": tokens[:train_end],
        "validation": tokens[train_end:validation_end],
        "test": tokens[validation_end:],
    }
    loaders = {
        name: CharacterBatchLoader(values, batch_size, sequence_length)
        for name, values in split_tokens.items()
    }
    return ShakespeareData(
        train=loaders["train"],
        validation=loaders["validation"],
        test=loaders["test"],
        vocabulary=vocabulary,
        split_sizes={name: len(values) for name, values in split_tokens.items()},
    )

