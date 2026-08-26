"""A character-level Elman RNN language model."""

from __future__ import annotations

import torch
from torch import nn


class CharacterRNN(nn.Module):
    """Predict the next character from a recurrent hidden state."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must be greater than 1")
        if embedding_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("model dimensions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity="tanh",
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, sequence]")
        embeddings = self.embedding(tokens)
        states, hidden = self.rnn(embeddings, hidden)
        return self.output(states), hidden

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively sample characters after a one-dimensional prompt."""
        if prompt_tokens.ndim != 1 or prompt_tokens.numel() == 0:
            raise ValueError("prompt_tokens must be a non-empty one-dimensional tensor")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self.eval()
        device = next(self.parameters()).device
        generated = prompt_tokens.to(device).unsqueeze(0)
        logits, hidden = self(generated)
        next_logits = logits[:, -1, :]

        for _ in range(max_new_tokens):
            probabilities = torch.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
            logits, hidden = self(next_token, hidden)
            next_logits = logits[:, -1, :]

        return generated.squeeze(0).cpu()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

