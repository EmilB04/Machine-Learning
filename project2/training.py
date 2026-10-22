"""Reusable training function for the Project 2 experiments (E2-E4).

train.py trains one fixed baseline. This module trains any configuration
with any seed and returns the full history, so experiments.py can run many
configurations and save the results. The test set is never used here:
all model selection is based on the validation set.

Verify against train.py (should print identical numbers):
    python train.py
    python training.py --baseline
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from mlp import ConfigurableMLP
from preprocessing import DataSplits, prepare_data


@dataclass
class TrainConfig:
    """All settings that define one training run (except the seed)."""

    hidden_sizes: tuple = (64, 32)
    activation: str = "relu"            # "relu", "sigmoid" or "tanh"
    dropout: float = 0.0
    optimizer: str = "adam"             # "adam", "adamw" or "sgd"
    learning_rate: float = 1e-3
    momentum: float = 0.0               # only used by SGD
    weight_decay: float = 0.0
    scheduler: str = "none"             # "none" or "cosine"
    batch_size: int = 64
    max_epochs: int = 200
    patience: int | None = 20           # None disables early stopping
    pos_weight: float | None = None     # positive-class weight in the loss (E5)


# Exactly the settings used in train.py, for verification.
BASELINE = TrainConfig(max_epochs=20, patience=5)


def make_optimizer(config: TrainConfig, parameters):
    if config.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unknown optimizer '{config.optimizer}'.")


def train_model(config: TrainConfig, splits: DataSplits, seed: int = 42, verbose: bool = False) -> dict:
    """Train one network and return its history and best validation result.

    The steps follow train.py in the same order (seed, model, DataLoader),
    so BASELINE with seed 42 reproduces train.py exactly.
    """
    start_time = time.perf_counter()

    # 1. Seed first: it decides the initial weights and the batch order.
    torch.manual_seed(seed)

    # 2. Model.
    input_dim = splits.X_train.shape[1]
    model = ConfigurableMLP(
        input_dim=input_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
        dropout=config.dropout,
    )

    # 3. Tensors. Labels are reshaped to (N, 1) to match the model output.
    X_train = torch.tensor(splits.X_train, dtype=torch.float32)
    y_train = torch.tensor(splits.y_train, dtype=torch.float32).reshape(-1, 1)
    X_validation = torch.tensor(splits.X_validation, dtype=torch.float32)
    y_validation = torch.tensor(splits.y_validation, dtype=torch.float32).reshape(-1, 1)

    # 4. Mini-batches, reshuffled every epoch.
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    # train.py draws one batch for inspection before training. That uses the
    # random number generator once, so we do the same to keep runs identical.
    next(iter(train_loader))

    # 5. Loss and optimizer. The training loss may weight the positive class;
    # the validation loss is always unweighted so runs stay comparable.
    if config.pos_weight is None:
        train_criterion = torch.nn.BCEWithLogitsLoss()
    else:
        train_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.pos_weight]))
    validation_criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = make_optimizer(config, model.parameters())
    scheduler = None
    if config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    elif config.scheduler != "none":
        raise ValueError(f"Unknown scheduler '{config.scheduler}'.")

    linear_layers = model.linear_layers()
    history = {
        "train_loss": [],        # running average over batches, as printed by train.py
        "train_loss_eval": [],   # full training set in eval mode (no dropout), for overfitting curves
        "validation_loss": [],
        "validation_ap": [],
        "gradient_norms": [],    # per epoch: mean L2 norm of each Linear layer's weight gradient
    }

    best_validation_ap = float("-inf")
    best_epoch = 0
    best_model_weights = None
    epochs_without_improvement = 0
    y_true = y_validation.numpy().ravel()

    # 6. Training loop.
    for epoch in range(config.max_epochs):
        model.train()
        total_loss = 0.0
        gradient_sums = np.zeros(len(linear_layers))
        batches = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = train_criterion(logits, y_batch)
            loss.backward()

            # Gradient size per layer, measured before the update.
            for i, layer in enumerate(linear_layers):
                gradient_sums[i] += layer.weight.grad.norm().item()
            batches += 1

            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        if scheduler is not None:
            scheduler.step()

        average_loss = total_loss / len(train_dataset)

        # 7. Evaluate after each epoch (no gradients, no dropout).
        model.eval()
        with torch.no_grad():
            train_loss_eval = validation_criterion(model(X_train), y_train).item()
            validation_logits = model(X_validation)
            validation_loss = validation_criterion(validation_logits, y_validation).item()
            y_scores = torch.sigmoid(validation_logits).numpy().ravel()
        validation_ap = average_precision_score(y_true, y_scores)

        # 8. Keep the weights from the epoch with the highest validation AP.
        if validation_ap > best_validation_ap:
            best_validation_ap = validation_ap
            best_epoch = epoch + 1
            best_model_weights = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history["train_loss"].append(average_loss)
        history["train_loss_eval"].append(train_loss_eval)
        history["validation_loss"].append(validation_loss)
        history["validation_ap"].append(validation_ap)
        history["gradient_norms"].append((gradient_sums / batches).tolist())

        if verbose:
            print(
                f"Epoch {epoch + 1:02d}/{config.max_epochs} | "
                f"Train loss: {average_loss:.4f} | "
                f"Val loss: {validation_loss:.4f} | "
                f"Val AP: {validation_ap:.4f}"
            )

        # 9. Early stopping.
        if config.patience is not None and epochs_without_improvement >= config.patience:
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    # 10. Restore the best weights.
    model.load_state_dict(best_model_weights)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_ap": best_validation_ap,
        "best_validation_loss": history["validation_loss"][best_epoch - 1],
        "epochs_run": len(history["validation_ap"]),
        "n_parameters": model.count_parameters(),
        "seconds": time.perf_counter() - start_time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one MLP configuration on the validation split.")
    parser.add_argument("--baseline", action="store_true", help="use exactly the settings from train.py")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = BASELINE if args.baseline else TrainConfig()
    splits = prepare_data()
    print(f"Config: {config}")
    print(f"Seed: {args.seed}\n")

    result = train_model(config, splits, seed=args.seed, verbose=True)

    print("\nBest model:")
    print("Best epoch:", result["best_epoch"])
    print("Best validation AP:", result["best_validation_ap"])
    print("Validation loss at best epoch:", result["best_validation_loss"])
    print("Trainable parameters:", result["n_parameters"])
    print(f"Training time: {result['seconds']:.1f} s")


if __name__ == "__main__":
    main()
