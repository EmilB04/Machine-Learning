
"""Final evaluation of the selected MLP model.

The final model is trained on the combined training and
validation sets, then evaluated once on the held-out test set.

All model settings must be finalized before test evaluation.
"""

import json
from math import sqrt
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlp import ConfigurableMLP
from preprocessing import prepare_data, prepare_final_data


# Project directories
PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"

# Final training settings
FINAL_EPOCHS = 40
CLASSIFICATION_THRESHOLD = 0.5
POS_WEIGHT = 1.0
RANDOM_SEED = 42


def load_final_preprocessing():
    """Load the input representation selected in E0."""

    path = RESULTS_DIR / "e0_selected.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing E0 configuration: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        selected = json.load(file)

    log_features = selected["log_features"]

    print("Selected preprocessing:", selected["name"])
    print("Log-transformed features:", log_features)

    return log_features


def load_model_config():
    """Load the MLP configuration selected in E4."""

    path = RESULTS_DIR / "e4_selected.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing model configuration: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        selected = json.load(file)

    config = selected["config"]

    print("Selected model configuration:")
    print("Hidden layers:", config["hidden_sizes"])
    print("Activation:", config["activation"])
    print("Optimizer:", config["optimizer"])
    print("Learning rate:", config["learning_rate"])
    print("Weight decay:", config["weight_decay"])

    return config


def prepare_final_dataset(log_features):
    """Prepare final training and test data.

    Preprocessing is fitted on train + validation.
    The test set remains held out.
    """

    # Create the original split using selected input coding.
    splits = prepare_data(
        log_features=log_features
    )

    # Combine training and validation.
    # Preprocessing options are inherited from splits.
    final_data = prepare_final_data(splits)

    print("\n=== FINAL DATASET ===")
    print("Training samples:", len(final_data.y_train))
    print("Test samples:", len(final_data.y_test))
    print("Input features:", final_data.X_train.shape[1])

    return final_data


def train_final_model(final_data, model_config):
    """Train the final MLP on train + validation.

    Uses a fixed number of epochs, without early stopping.
    No test data is used during training.
    """

    torch.manual_seed(RANDOM_SEED)

    # Convert training data to PyTorch tensors.
    X_train = torch.tensor(
        final_data.X_train,
        dtype=torch.float32,
    )

    y_train = torch.tensor(
        final_data.y_train,
        dtype=torch.float32,
    ).reshape(-1, 1)

    # Create mini-batches.
    dataset = TensorDataset(X_train, y_train)

    loader = DataLoader(
        dataset,
        batch_size=model_config["batch_size"],
        shuffle=True,
    )

    # Build the selected MLP architecture.
    model = ConfigurableMLP(
        input_dim=X_train.shape[1],
        hidden_sizes=tuple(model_config["hidden_sizes"]),
        activation=model_config["activation"],
        dropout=model_config["dropout"],
    )

    # Binary classification loss.
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT])
    )

    # Selected optimizer.
    if model_config["optimizer"] != "sgd":
        raise ValueError(
            "Expected SGD. Check the final model selection."
        )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=model_config["learning_rate"],
        momentum=model_config["momentum"],
        weight_decay=model_config["weight_decay"],
    )

    print("\n=== FINAL MODEL TRAINING ===")
    print("Architecture:", model_config["hidden_sizes"])
    print("Trainable parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    # Train for the predefined number of epochs.
    for epoch in range(FINAL_EPOCHS):

        model.train()
        total_loss = 0.0

        for X_batch, y_batch in loader:

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = criterion(logits, y_batch)

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * X_batch.size(0)
            )

        average_loss = total_loss / len(dataset)

        print(
            f"Epoch {epoch + 1:02d}/{FINAL_EPOCHS} | "
            f"Training loss: {average_loss:.4f}"
        )

    model.eval()

    print("\nFinal training completed.")
    print("No test evaluation performed.")

    return model


def wilson_confidence_interval(errors, total, z=1.96):
    """Calculate a 95% Wilson confidence interval.

    Args:
        errors: Number of misclassified observations.
        total: Total number of test observations.
        z: Critical value for a 95% confidence level.

    Returns:
        Lower and upper confidence bounds.
    """

    if total <= 0:
        raise ValueError("Total must be positive.")

    if not 0 <= errors <= total:
        raise ValueError("Invalid number of errors.")

    p = errors / total

    denominator = 1 + z**2 / total

    center = (
        p + z**2 / (2 * total)
    ) / denominator

    margin = (
        z
        * sqrt(
            p * (1 - p) / total
            + z**2 / (4 * total**2)
        )
        / denominator
    )

    return center - margin, center + margin


def evaluate_final_model(model, final_data):
    """Evaluate the final model on the held-out test set.

    IMPORTANT:
    Only call after all model settings are frozen.
    """

    model.eval()

    X_test = torch.tensor(
        final_data.X_test,
        dtype=torch.float32,
    )

    y_true = final_data.y_test

    # Calculate predicted probabilities.
    with torch.no_grad():
        logits = model(X_test)

        probabilities = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
            .ravel()
        )

    # Convert probabilities to class predictions.
    predictions = (
        probabilities >= CLASSIFICATION_THRESHOLD
    ).astype(int)

    # Classification metrics.
    ap = average_precision_score(
        y_true, probabilities
    )

    roc_auc = roc_auc_score(
        y_true, probabilities
    )

    accuracy = accuracy_score(
        y_true, predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true, predictions
    )

    precision = precision_score(
        y_true, predictions, zero_division=0
    )

    recall = recall_score(
        y_true, predictions, zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, zero_division=0
    )

    # Confusion matrix.
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    # Error rate and confidence interval.
    errors = int(fp + fn)
    total = len(y_true)

    error_rate = errors / total

    ci_lower, ci_upper = wilson_confidence_interval(
        errors,
        total,
    )

    results = {
        "AP": float(ap),
        "ROC_AUC": float(roc_auc),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "error_rate": float(error_rate),
        "error_CI_lower": float(ci_lower),
        "error_CI_upper": float(ci_upper),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "test_samples": int(total),
    }

    print("\n=== FINAL TEST RESULTS ===")

    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    return results, y_true, probabilities, predictions


def save_final_results(
    results,
    y_true,
    probabilities,
    predictions,
):
    """Save final metrics and evaluation figures."""

    output_dir = RESULTS_DIR / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save classification metrics.
    metrics_df = pd.DataFrame([results])

    metrics_df.to_csv(
        output_dir / "final_metrics.csv",
        index=False,
    )

    # Confusion matrix.
    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["No purchase", "Purchase"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )

    ax.set_title("Final MLP: Confusion Matrix")

    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Precision-recall curve.
    fig, ax = plt.subplots(figsize=(6, 5))

    PrecisionRecallDisplay.from_predictions(
        y_true,
        probabilities,
        ax=ax,
        name="MLP",
    )

    ax.set_title("Final MLP: Precision-Recall Curve")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        output_dir / "precision_recall_curve.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n=== RESULTS SAVED ===")
    print("Output directory:", output_dir)


def main():
    print("=== FINAL MODEL CONFIGURATION ===")

    log_features = load_final_preprocessing()
    model_config = load_model_config()
    final_data = prepare_final_dataset(log_features)

    print("\nFinal training settings:")
    print("Epochs:", FINAL_EPOCHS)
    print("Positive class weight:", POS_WEIGHT)
    print("Classification threshold:", CLASSIFICATION_THRESHOLD)

    print("\nConfiguration loaded successfully.")
    print("No model has been trained or evaluated.")


if __name__ == "__main__":
    main()
