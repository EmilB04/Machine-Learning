
"""Final evaluation of the selected MLP model.

The final model is trained on the combined training and
validation sets, then evaluated once on the held-out test set.

All model settings must be finalized before test evaluation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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


def main():
    print("=== FINAL MODEL CONFIGURATION ===")

    log_features = load_final_preprocessing()
    model_config = load_model_config()

    print("\nFinal training settings:")
    print("Epochs:", FINAL_EPOCHS)
    print("Positive class weight:", POS_WEIGHT)
    print("Classification threshold:", CLASSIFICATION_THRESHOLD)

    print("\nConfiguration loaded successfully.")
    print("No model has been trained or evaluated.")


if __name__ == "__main__":
    main()
