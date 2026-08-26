"""Train, evaluate, and save the Character RNN."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from model.rnn import CharacterRNN
from sample import sample_text
from util.config import get_device, seed_everything
from util.data_loader import CharacterBatchLoader, ShakespeareData, get_shakespeare_data


def run_epoch(
    model: CharacterRNN,
    loader: CharacterBatchLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> dict[str, float]:
    """Process continuous streams, carrying hidden state between sequence blocks."""
    training = optimizer is not None
    model.train(training)
    hidden: torch.Tensor | None = None
    total_nll = 0.0
    total_characters = 0
    context = torch.enable_grad if training else torch.inference_mode

    with context():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            if hidden is not None:
                hidden = hidden.detach()
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, hidden = model(inputs, hidden)
            loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size), targets.reshape(-1)
            )
            if training:
                loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            characters = targets.numel()
            total_nll += loss.item() * characters
            total_characters += characters

    mean_nll = total_nll / total_characters
    return {"nll": mean_nll, "perplexity": math.exp(mean_nll)}


def _copy_state_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def run_pipeline(config: dict) -> None:
    """Run data loading, training, evaluation, and text generation."""
    seed_everything(config["seed"])
    device = get_device()
    data: ShakespeareData = get_shakespeare_data(
        config["data_dir"],
        sequence_length=config["sequence_length"],
        batch_size=config["batch_size"],
        train_fraction=config["train_fraction"],
        validation_fraction=config["validation_fraction"],
    )
    model = CharacterRNN(
        vocab_size=len(data.vocabulary),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    sizes = data.split_sizes
    print(f"device: {device}")
    print(
        f"characters: train {sizes['train']:,} | validation "
        f"{sizes['validation']:,} | test {sizes['test']:,}"
    )
    print(f"vocabulary: {len(data.vocabulary)} | parameters: {model.parameter_count():,}")

    best_validation_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(
            model, data.train, device, optimizer, config["grad_clip"]
        )
        validation_metrics = run_epoch(model, data.validation, device)
        if validation_metrics["nll"] < best_validation_nll:
            best_validation_nll = validation_metrics["nll"]
            best_state = _copy_state_to_cpu(model)
        print(
            f"epoch {epoch:02d}/{config['epochs']} | "
            f"train NLL {train_metrics['nll']:.4f} | "
            f"validation NLL {validation_metrics['nll']:.4f} | "
            f"validation perplexity {validation_metrics['perplexity']:.2f}"
        )

    if best_state is None:
        raise RuntimeError("training did not produce a model state")
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = run_epoch(model, data.test, device)
    print(
        f"test NLL {test_metrics['nll']:.4f} | "
        f"test perplexity {test_metrics['perplexity']:.2f}"
    )

    checkpoint = {
        "model_state_dict": best_state,
        "vocabulary": data.vocabulary.itos,
        "config": config,
        "validation_nll": best_validation_nll,
        "test_nll": test_metrics["nll"],
    }
    torch.save(checkpoint, "rnn_model.pt")

    generated_text = sample_text(
        model,
        data.vocabulary,
        prompt=str(config["prompt"]),
        length=config["generation_length"],
        temperature=config["temperature"],
        seed=config["seed"] + 10_000,
    )
    Path("generated_text.txt").write_text(generated_text, encoding="utf-8")
    print("model: rnn_model.pt")
    print("generated text: generated_text.txt")
    print("\n--- sample ---\n")
    print(generated_text)

