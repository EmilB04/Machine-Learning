"""Run Project 2 data checks and prepare MLP inputs."""

from preprocessing import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_data,
    prepare_final_data,
)


def section(title: str) -> None:
    print(f"\n{title}\n{'─' * len(title)}")


def number(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def print_partition(label: str, y_values) -> None:
    purchases = int(y_values.sum())
    total = len(y_values)
    non_purchases = total - purchases
    purchase_rate = purchases / total * 100
    print(
        f"{label:<20} {number(total):>8}  "
        f"{number(non_purchases):>12}  {number(purchases):>10}  {purchase_rate:>6.1f}%"
    )


def main() -> None:
    splits = prepare_data()
    audit = splits.audit
    total = int(audit["rows"])
    duplicates = int(audit["exact_duplicate_rows"])
    conflicts = int(audit["conflicting_label_combinations"])
    cross_split_duplicates = int(audit["test_rows_identical_to_training_rows"])

    print("╭────────────────────────────────────────────╮")
    print("│  PROJECT 2  /  DATA PREPARATION             │")
    print("╰────────────────────────────────────────────╯")
    print(f"Dataset  {number(total)} rows  ·  {audit['predictors']} predictors  ·  Revenue prediction")

    section("DATA QUALITY")
    print(f"  Missing values                 {number(int(audit['missing_values'])):>8}")
    print(f"  Exact duplicate rows           {number(duplicates):>8}")
    print(f"  Conflicting label combinations {number(conflicts):>8}")
    print(f"  Identical rows across train/test{number(cross_split_duplicates):>8}")

    class_counts = audit["target_counts"]
    class_total = sum(class_counts.values())
    print("\n  Revenue distribution")
    for label, description in ((0, "No purchase"), (1, "Purchase")):
        count = int(class_counts.get(label, 0))
        share = count / class_total * 100
        print(f"    {description:<14} {number(count):>8}  ({share:5.1f}%)")

    section("MODEL INPUTS")
    print(f"  Categorical  {len(CATEGORICAL_FEATURES):>2} columns  →  one-hot encoding")
    print(f"  Numeric      {len(NUMERIC_FEATURES):>2} columns  →  standard scaling")
    print(f"  Binary       {len(BINARY_FEATURES):>2} column   →  passed through as 0/1")
    print(f"  Encoded features for tuning: {len(splits.feature_names)}")
    print("  Preprocessing is fitted on training data only during model selection.")

    section("STRATIFIED SPLIT  ·  60 / 20 / 20")
    print(f"{'Partition':<20} {'Rows':>8}  {'No purchase':>12}  {'Purchase':>10}  {'Rate':>7}")
    print(f"{'─' * 20} {'─' * 8}  {'─' * 12}  {'─' * 10}  {'─' * 7}")
    print_partition("Train", splits.y_train)
    print_partition("Validation", splits.y_validation)
    print_partition("Held-out test", splits.y_test)

    final_data = prepare_final_data(splits)
    section("FINAL FIT  ·  AFTER MODEL CHOICES ARE FROZEN")
    print_partition("Train + validation", final_data.y_train)
    print_partition("Held-out test", final_data.y_test)
    print(f"  Final encoded input features: {len(final_data.feature_names)}")
    print("  Refit preprocessing on all 80% training rows, then initialize and train a fresh MLP.")

    if duplicates or conflicts or cross_split_duplicates:
        section("CHECK NEEDED")
        if duplicates:
            print(f"  ⚠ {number(duplicates)} exact duplicate rows remain in the input.")
        if conflicts:
            print(f"  ⚠ {number(conflicts)} predictor combinations have conflicting labels.")
        if cross_split_duplicates:
            print(f"  ⚠ {number(cross_split_duplicates)} identical full rows cross train/test.")
    else:
        print("\n✓ Data checks passed; no duplicate rows cross the train/test boundary.")


if __name__ == "__main__":
    main()
