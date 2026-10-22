"""Figures for the Project 2 report, made from the saved experiment results.

No training happens here: everything is read from results/*.csv and
results/*_histories.json, so the script runs in a few seconds.

    python plots.py

Writes PDF (for LaTeX) and PNG (for quick viewing) to results/figures/:
    fig_e3_effective_lr      E3: validation AP vs effective learning rate, SGD vs Adam
    fig_overfitting          E4 run without early stopping: loss and AP per epoch
    fig_vanishing_gradients  Diag: gradient norm per layer and AP per epoch, sigmoid vs ReLU
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Colours from a colour-blind-checked categorical palette, always used in this
# order. Identity is never colour alone: every series also has its own marker
# or line style and a legend entry.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GREY = "#6b6b6b"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e4e4e4",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.8,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})


def load_histories(stage):
    with open(RESULTS_DIR / f"{stage}_histories.json", encoding="utf-8") as file:
        return json.load(file)


def stack(histories, config, key):
    """Return an array (seeds x epochs) of one history key for one configuration."""
    runs = [h[key] for name, h in histories.items() if name.startswith(config + "|")]
    if not runs:
        raise KeyError(f"No runs for '{config}' in the histories file.")
    return np.array(runs)


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{suffix}")
    plt.close(fig)
    print(f"Saved results/figures/{name}.pdf and .png")


# --------------------------------------------------------------------------
# E3: validation AP against effective learning rate lr / (1 - momentum)
# --------------------------------------------------------------------------

def plot_e3_effective_lr():
    runs = pd.read_csv(RESULTS_DIR / "e3_runs.csv")
    runs = runs[runs["scheduler"] == "none"]  # the cosine variant is a separate point in the text
    max_epochs = runs["epochs_run"].max()

    sgd = runs[runs["optimizer"] == "sgd"].copy()
    sgd["effective_lr"] = sgd["learning_rate"] / (1 - sgd["momentum"])
    # A configuration is cap-limited if any seed ran to the epoch cap without
    # early stopping being triggered: its AP is then a lower bound.
    grouped = (
        sgd.groupby(["momentum", "learning_rate"])
        .agg(
            effective_lr=("effective_lr", "first"),
            mean_ap=("best_validation_ap", "mean"),
            sd_ap=("best_validation_ap", "std"),
            cap_limited=("epochs_run", lambda e: bool((e >= max_epochs).any())),
        )
        .reset_index()
    )

    adam = runs[runs["optimizer"] == "adam"]
    best_adam = adam.groupby("learning_rate")["best_validation_ap"].mean()

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    styles = [(0.0, BLUE, "o"), (0.5, ORANGE, "s"), (0.9, AQUA, "^")]
    for momentum, colour, marker in styles:
        part = grouped[grouped["momentum"] == momentum].sort_values("effective_lr")
        ax.plot(part["effective_lr"], part["mean_ap"], color=colour, linewidth=1.2, zorder=2)
        ax.errorbar(
            part["effective_lr"], part["mean_ap"], yerr=part["sd_ap"],
            fmt="none", ecolor=colour, elinewidth=1, capsize=2, zorder=2,
        )
        converged = part[~part["cap_limited"]]
        capped = part[part["cap_limited"]]
        ax.scatter(converged["effective_lr"], converged["mean_ap"], color=colour, marker=marker,
                   s=36, edgecolor="white", linewidth=0.8, zorder=3, label=f"SGD, momentum {momentum:g}")
        ax.scatter(capped["effective_lr"], capped["mean_ap"], facecolor="white", edgecolor=colour,
                   marker=marker, s=36, linewidth=1.2, zorder=3)

    ax.axhline(best_adam.max(), color=GREY, linestyle="--", linewidth=1.2, zorder=1,
               label=f"Adam, best lr ({best_adam.idxmax():g})")
    # Legend entry explaining the hollow markers.
    ax.scatter([], [], facecolor="white", edgecolor=GREY, marker="o", s=36, linewidth=1.2,
               label=f"hit the {max_epochs}-epoch cap (lower bound)")

    ax.set_xscale("log")
    ax.set_xlabel("Effective learning rate, lr / (1 − momentum)")
    ax.set_ylabel("Validation AP (mean ± SD, 5 seeds)")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "fig_e3_effective_lr")


# --------------------------------------------------------------------------
# E4: the selected configuration trained for 200 epochs without early stopping
# --------------------------------------------------------------------------

def plot_overfitting():
    histories = load_histories("e4_no_early_stopping")
    config = "no-early-stopping"
    train_loss = stack(histories, config, "train_loss_eval")
    val_loss = stack(histories, config, "validation_loss")
    val_ap = stack(histories, config, "validation_ap")
    epochs = np.arange(1, train_loss.shape[1] + 1)
    best_epoch = float(np.mean(np.argmax(val_ap, axis=1) + 1))

    fig, (ax_loss, ax_ap) = plt.subplots(1, 2, figsize=(6.6, 2.7))

    for values, colour, style, label in (
        (train_loss, BLUE, "-", "Training loss"),
        (val_loss, ORANGE, "--", "Validation loss"),
    ):
        ax_loss.fill_between(epochs, values.min(axis=0), values.max(axis=0), color=colour, alpha=0.15, linewidth=0)
        ax_loss.plot(epochs, values.mean(axis=0), color=colour, linestyle=style, label=label)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Binary cross-entropy")
    ax_loss.set_title("(a) Loss")
    ax_loss.legend(frameon=False)

    ax_ap.fill_between(epochs, val_ap.min(axis=0), val_ap.max(axis=0), color=ORANGE, alpha=0.15, linewidth=0)
    ax_ap.plot(epochs, val_ap.mean(axis=0), color=ORANGE, linewidth=1.4, label="Validation AP")
    ax_ap.set_xlabel("Epoch")
    ax_ap.set_ylabel("Validation AP")
    ax_ap.set_title("(b) Validation AP")

    for ax in (ax_loss, ax_ap):
        ax.axvline(best_epoch, color=GREY, linestyle=":", linewidth=1.2)
    ax_ap.annotate(f"mean best epoch ≈ {best_epoch:.0f}", xy=(best_epoch, ax_ap.get_ylim()[0]),
                   xytext=(6, 6), textcoords="offset points", fontsize=8, color=GREY)

    fig.tight_layout()
    save(fig, "fig_overfitting")


# --------------------------------------------------------------------------
# Diag: vanishing gradients in a deep sigmoid network vs a deep ReLU network
# --------------------------------------------------------------------------

def plot_vanishing_gradients():
    histories = load_histories("diag")
    fig, (ax_grad, ax_ap) = plt.subplots(1, 2, figsize=(6.6, 2.7))

    for config, colour, marker, style, label in (
        ("deep6-sigmoid", ORANGE, "s", "-", "Sigmoid"),
        ("deep6-relu", BLUE, "o", "--", "ReLU"),
    ):
        norms = stack(histories, config, "gradient_norms")  # seeds x epochs x layers
        first_epoch = norms[:, 0, :].mean(axis=0)
        layers = np.arange(1, len(first_epoch) + 1)
        ax_grad.plot(layers, first_epoch, color=colour, linestyle=style, marker=marker,
                     markersize=6, markeredgecolor="white", label=label)

        ap = stack(histories, config, "validation_ap")
        epochs = np.arange(1, ap.shape[1] + 1)
        ax_ap.fill_between(epochs, ap.min(axis=0), ap.max(axis=0), color=colour, alpha=0.15, linewidth=0)
        ax_ap.plot(epochs, ap.mean(axis=0), color=colour, linestyle=style, label=label)

    ax_grad.set_yscale("log")
    ax_grad.set_xticks(layers)
    ax_grad.set_xticklabels([str(i) for i in layers[:-1]] + ["out"])
    ax_grad.set_xlabel("Layer (1 = closest to input)")
    ax_grad.set_ylabel("Weight-gradient norm (log scale)")
    ax_grad.set_title("(a) Gradient norm per layer, epoch 1")
    ax_grad.legend(frameon=False)

    ax_ap.set_xlabel("Epoch")
    ax_ap.set_ylabel("Validation AP")
    ax_ap.set_title("(b) Validation AP per epoch")
    ax_ap.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    save(fig, "fig_vanishing_gradients")


def main():
    plot_e3_effective_lr()
    plot_overfitting()
    plot_vanishing_gradients()


if __name__ == "__main__":
    main()
