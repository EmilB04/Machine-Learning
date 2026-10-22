"""Run experiments E2-E4 and the gradient diagnostics for Project 2.

Every configuration is trained with five seeds (0-4). All selection uses the
validation set; the test set is never touched here.

Stages (each builds on the winner of the previous one):
    e2    Architecture: 1-3 hidden layers x widths 16/32/64/128
    e3    Optimizer: Adam vs SGD with momentum, several learning rates, cosine schedule
    e4    Regularization: dropout x weight decay, plus one run without early stopping
    e4curve  Only the run without early stopping, for the configuration selected in E4
    diag  Vanishing gradients: deep sigmoid network vs deep ReLU network

Usage:
    python experiments.py --stage e2 --quick   # smoke test, about a minute
    python experiments.py --stage e2
    python experiments.py --stage e3
    python experiments.py --stage e4
    python experiments.py --stage diag

Output per stage in results/ (results/quick/ with --quick):
    <stage>_runs.csv        one row per configuration and seed
    <stage>_summary.csv     mean and standard deviation over seeds
    <stage>_histories.json  per-epoch losses, AP and gradient norms
    <stage>_selected.json   the configuration carried forward (e2, e3, e4)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from preprocessing import prepare_data
from training import TrainConfig, train_model


RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEEDS = [0, 1, 2, 3, 4]

# Shared starting point: the baseline settings, but with room to train longer.
# SGD learns more slowly than Adam, so a cap of 20 epochs would be unfair to it.
BASE = TrainConfig(max_epochs=200, patience=20)


# --------------------------------------------------------------------------
# Running and saving
# --------------------------------------------------------------------------

def run_stage(stage, named_configs, splits, seeds, output_dir):
    """Train every (name, config) pair with every seed and save the results."""
    rows = []
    histories = {}

    total = len(named_configs) * len(seeds)
    done = 0
    for name, config in named_configs:
        for seed in seeds:
            result = train_model(config, splits, seed=seed)
            done += 1
            print(
                f"[{stage}] {done:>3}/{total}  {name:<24} seed {seed}  "
                f"AP {result['best_validation_ap']:.4f}  "
                f"best epoch {result['best_epoch']:>3}/{result['epochs_run']:<3}  "
                f"{result['seconds']:5.1f} s"
            )

            rows.append({
                "stage": stage,
                "name": name,
                "seed": seed,
                "hidden_sizes": "-".join(str(w) for w in config.hidden_sizes),
                "activation": config.activation,
                "dropout": config.dropout,
                "optimizer": config.optimizer,
                "learning_rate": config.learning_rate,
                "momentum": config.momentum,
                "weight_decay": config.weight_decay,
                "scheduler": config.scheduler,
                "patience": config.patience,
                "best_validation_ap": result["best_validation_ap"],
                "best_validation_loss": result["best_validation_loss"],
                "best_epoch": result["best_epoch"],
                "epochs_run": result["epochs_run"],
                "n_parameters": result["n_parameters"],
                "seconds": result["seconds"],
            })
            histories[f"{name}|seed{seed}"] = result["history"]

        # Save after every configuration, so a crash does not lose finished runs.
        save_stage(stage, rows, histories, output_dir)

    return summarize(pd.DataFrame(rows))


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds, one row per configuration."""
    summary = (
        runs.groupby("name", sort=False)
        .agg(
            mean_ap=("best_validation_ap", "mean"),
            sd_ap=("best_validation_ap", "std"),
            min_ap=("best_validation_ap", "min"),
            max_ap=("best_validation_ap", "max"),
            mean_val_loss=("best_validation_loss", "mean"),
            mean_best_epoch=("best_epoch", "mean"),
            mean_epochs_run=("epochs_run", "mean"),
            n_parameters=("n_parameters", "first"),
            mean_seconds=("seconds", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values("mean_ap", ascending=False).reset_index(drop=True)


def save_stage(stage, rows, histories, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(output_dir / f"{stage}_runs.csv", index=False)
    summarize(runs).to_csv(output_dir / f"{stage}_summary.csv", index=False)
    with open(output_dir / f"{stage}_histories.json", "w", encoding="utf-8") as file:
        json.dump(histories, file)


def save_selected(stage, name, config, summary, output_dir, reason):
    row = summary.loc[summary["name"] == name].iloc[0]
    payload = {
        "stage": stage,
        "name": name,
        "reason": reason,
        "mean_ap": float(row["mean_ap"]),
        "sd_ap": float(row["sd_ap"]),
        "config": asdict(config),
    }
    with open(output_dir / f"{stage}_selected.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"\n[{stage}] Selected: {name}  (mean AP {row['mean_ap']:.4f} ± {row['sd_ap']:.4f})")
    print(f"[{stage}] Reason: {reason}")


def load_selected(stage, output_dir) -> TrainConfig:
    path = output_dir / f"{stage}_selected.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run --stage {stage} first.")
    with open(path, encoding="utf-8") as file:
        config = json.load(file)["config"]
    config["hidden_sizes"] = tuple(config["hidden_sizes"])
    return TrainConfig(**config)


def print_summary(stage, summary):
    print(f"\n[{stage}] Summary (validation AP over seeds, best first):")
    columns = ["name", "mean_ap", "sd_ap", "mean_best_epoch", "mean_epochs_run", "n_parameters"]
    print(summary[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# --------------------------------------------------------------------------
# Selection rules (fixed before looking at the results)
# --------------------------------------------------------------------------

def select_simplest_within_one_sd(summary):
    """E2: among configurations whose mean AP is within one standard deviation
    of the best one, choose the one with the fewest parameters."""
    best = summary.iloc[0]
    threshold = best["mean_ap"] - best["sd_ap"]
    candidates = summary[summary["mean_ap"] >= threshold]
    chosen = candidates.sort_values(["n_parameters", "mean_ap"], ascending=[True, False]).iloc[0]
    reason = (
        f"fewest parameters among {len(candidates)} configuration(s) with mean AP "
        f">= {threshold:.4f} (best mean minus one SD of {best['name']})"
    )
    return chosen["name"], reason


def select_highest_mean(summary):
    """E3 and E4: highest mean validation AP."""
    return summary.iloc[0]["name"], "highest mean validation AP over seeds"


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def stage_e2(splits, seeds, output_dir, base):
    named = []
    for depth in (1, 2, 3):
        for width in (16, 32, 64, 128):
            # Halve the width for each extra layer, e.g. 64 -> (64, 32, 16).
            hidden = tuple(width // (2 ** i) for i in range(depth))
            named.append((f"L{depth}-W{width}", replace(base, hidden_sizes=hidden)))

    summary = run_stage("e2", named, splits, seeds, output_dir)
    print_summary("e2", summary)
    name, reason = select_simplest_within_one_sd(summary)
    save_selected("e2", name, dict(named)[name], summary, output_dir, reason)


def stage_e3(splits, seeds, output_dir, base):
    architecture = load_selected("e2", output_dir)
    named = []
    for lr in (1e-4, 3e-4, 1e-3, 3e-3):
        named.append((f"Adam-lr{lr:g}", replace(architecture, optimizer="adam", learning_rate=lr)))
    for momentum in (0.0, 0.5, 0.9):
        for lr in (0.003, 0.01, 0.03, 0.1):
            named.append((
                f"SGD-m{momentum:g}-lr{lr:g}",
                replace(architecture, optimizer="sgd", momentum=momentum, learning_rate=lr),
            ))

    summary = run_stage("e3", named, splits, seeds, output_dir)

    # Cosine learning-rate schedule on the best optimizer setting found above.
    # The schedule decays the learning rate to zero over max_epochs, so early
    # stopping would cut it off before it takes effect. Instead it runs for the
    # number of epochs the best setting used on average, without early stopping,
    # and the weights from the epoch with the highest validation AP are kept.
    best_name = summary.iloc[0]["name"]
    best_config = dict(named)[best_name]
    budget = max(1, round(summary.iloc[0]["mean_epochs_run"]))
    cosine = replace(best_config, scheduler="cosine", patience=None, max_epochs=budget)
    named.append((f"{best_name}+cosine", cosine))
    summary = _extend_stage("e3", named[-1:], splits, seeds, output_dir)

    print_summary("e3", summary)
    name, reason = select_highest_mean(summary)
    save_selected("e3", name, dict(named)[name], summary, output_dir, reason)


def _extend_stage(stage, extra_configs, splits, seeds, output_dir):
    """Add configurations to a stage that has already been saved."""
    previous_runs = pd.read_csv(output_dir / f"{stage}_runs.csv")
    with open(output_dir / f"{stage}_histories.json", encoding="utf-8") as file:
        previous_histories = json.load(file)

    run_stage(f"{stage}_extra", extra_configs, splits, seeds, output_dir)
    extra_runs = pd.read_csv(output_dir / f"{stage}_extra_runs.csv")
    with open(output_dir / f"{stage}_extra_histories.json", encoding="utf-8") as file:
        extra_histories = json.load(file)

    extra_runs["stage"] = stage
    rows = pd.concat([previous_runs, extra_runs], ignore_index=True).to_dict("records")
    save_stage(stage, rows, {**previous_histories, **extra_histories}, output_dir)
    for suffix in ("runs.csv", "summary.csv", "histories.json"):
        (output_dir / f"{stage}_extra_{suffix}").unlink()
    return summarize(pd.DataFrame(rows))


def stage_e4(splits, seeds, output_dir, base):
    tuned = load_selected("e3", output_dir)
    named = []
    for dropout in (0.0, 0.2, 0.5):
        for weight_decay in (0.0, 1e-4, 1e-3):
            config = replace(tuned, dropout=dropout, weight_decay=weight_decay)
            # For Adam, weight decay is applied in decoupled form (AdamW), as
            # recommended by Loshchilov and Hutter; for SGD, both forms are equal.
            if config.optimizer == "adam" and weight_decay > 0:
                config = replace(config, optimizer="adamw")
            named.append((f"drop{dropout:g}-wd{weight_decay:g}", config))

    summary = run_stage("e4", named, splits, seeds, output_dir)
    print_summary("e4", summary)
    name, reason = select_highest_mean(summary)
    save_selected("e4", name, dict(named)[name], summary, output_dir, reason)

    stage_e4_curve(splits, seeds, output_dir, base)


def stage_e4_curve(splits, seeds, output_dir, base):
    # The configuration selected in E4, trained without early stopping, to show
    # the overfitting curve. Not part of the selection: it only documents what
    # happens after the best epoch. The reported AP is still the best epoch's AP
    # (best weights are restored), so the histories are what matter here.
    final = load_selected("e4", output_dir)
    no_stop = [("no-early-stopping", replace(final, patience=None, max_epochs=base.max_epochs))]
    summary_no_stop = run_stage("e4_no_early_stopping", no_stop, splits, seeds, output_dir)
    print_summary("e4_no_early_stopping", summary_no_stop)


def stage_diag(splits, seeds, output_dir, base):
    # Six hidden layers of 64 units, trained for a fixed number of epochs.
    # Compare the gradient norm of the first layer between sigmoid and ReLU.
    epochs = min(base.max_epochs, 100)
    deep = replace(base, hidden_sizes=(64,) * 6, patience=None, max_epochs=epochs)
    named = [
        ("deep6-sigmoid", replace(deep, activation="sigmoid")),
        ("deep6-relu", replace(deep, activation="relu")),
    ]
    summary = run_stage("diag", named, splits, seeds, output_dir)
    print_summary("diag", summary)


STAGES = {"e2": stage_e2, "e3": stage_e3, "e4": stage_e4, "e4curve": stage_e4_curve, "diag": stage_diag}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Project 2 MLP experiments on the validation split.")
    parser.add_argument("--stage", choices=[*STAGES, "all"], required=True)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke test: 2 seeds, at most 5 epochs, results in results/quick/",
    )
    args = parser.parse_args()

    seeds = SEEDS
    output_dir = RESULTS_DIR
    base = BASE
    if args.quick:
        seeds = SEEDS[:2]
        output_dir = RESULTS_DIR / "quick"
        base = replace(BASE, max_epochs=5, patience=3)

    splits = prepare_data()
    # e4 already runs e4curve at the end, so "all" does not list it separately.
    stages = ["e2", "e3", "e4", "diag"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"\n===== Stage {stage} =====")
        STAGES[stage](splits, seeds, output_dir, base)


if __name__ == "__main__":
    main()
