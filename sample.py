"""Generate text from a trained Character RNN."""

from __future__ import annotations

import torch

from model.rnn import CharacterRNN
from util.config import seed_everything
from util.data_loader import CharacterVocabulary


def sample_text(
    model: CharacterRNN,
    vocabulary: CharacterVocabulary,
    *,
    prompt: str,
    length: int,
    temperature: float,
    seed: int,
) -> str:
    """Warm up the RNN with a prompt and sample additional characters."""
    seed_everything(seed)
    prompt_tokens = vocabulary.encode(prompt)
    generated_tokens = model.generate(
        prompt_tokens,
        max_new_tokens=length,
        temperature=temperature,
    )
    return vocabulary.decode(generated_tokens)

