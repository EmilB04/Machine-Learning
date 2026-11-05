"""Data checks, splitting, and preprocessing for Project 2."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "online_shoppers_intention_deduplicated.csv"
)
TARGET = "Revenue"
RANDOM_STATE = 42

CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
]
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]
BINARY_FEATURES = ["Weekend"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + BINARY_FEATURES
EXPECTED_COLUMNS = set(FEATURES + [TARGET])

# E0: right-skewed, non-negative variables that get log1p before standardization.
LOG1P_FEATURES = [
    "PageValues",
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration",
]


@dataclass
class DataSplits:
    """Prepared arrays and metadata shared with the model-training code."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: pd.Index
    validation_indices: pd.Index
    test_indices: pd.Index
    feature_names: np.ndarray
    preprocessor: ColumnTransformer
    audit: dict[str, object]
    # Input-coding options used for this split (E0/E6); empty means the baseline.
    options: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class FinalData:
    """Final model inputs after model choices are frozen."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: pd.Index
    test_indices: pd.Index
    feature_names: np.ndarray
    preprocessor: ColumnTransformer


def load_data(data_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the original Project 1 CSV without modifying it."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path)
    if set(data.columns) != EXPECTED_COLUMNS:
        missing = sorted(EXPECTED_COLUMNS - set(data.columns))
        unexpected = sorted(set(data.columns) - EXPECTED_COLUMNS)
        raise ValueError(f"Unexpected dataset schema; missing={missing}, unexpected={unexpected}")

    return data[FEATURES + [TARGET]]


def audit_data(data: pd.DataFrame) -> dict[str, object]:
    """Summarize data quality and fail early for invalid model inputs."""
    missing_by_column = data.isna().sum()
    if missing_by_column.any():
        missing = missing_by_column[missing_by_column > 0].to_dict()
        raise ValueError(f"Missing values found: {missing}")

    if not set(data[TARGET].unique()).issubset({True, False, 0, 1}):
        raise ValueError("Revenue must contain only boolean or binary values.")

    conflicting_labels = (
        data.groupby(FEATURES, dropna=False, observed=True)[TARGET]
        .nunique()
        .gt(1)
        .sum()
    )
    if conflicting_labels:
        raise ValueError(f"Found {conflicting_labels} predictor combinations with conflicting Revenue labels.")

    duplicate_rows = int(data.duplicated().sum())
    target_counts = data[TARGET].astype(int).value_counts().sort_index().to_dict()
    target_proportions = data[TARGET].astype(int).value_counts(normalize=True).sort_index().to_dict()
    return {
        "rows": len(data),
        "predictors": len(FEATURES),
        "column_dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "missing_values": 0,
        "exact_duplicate_rows": duplicate_rows,
        "conflicting_label_combinations": int(conflicting_labels),
        "target_counts": {int(label): int(count) for label, count in target_counts.items()},
        "target_proportions": {
            int(label): float(proportion) for label, proportion in target_proportions.items()
        },
    }


def _check_options(log_features, drop_features) -> tuple[list[str], list[str]]:
    log_features = list(log_features or [])
    drop_features = list(drop_features or [])
    unknown_log = sorted(set(log_features) - set(NUMERIC_FEATURES))
    unknown_drop = sorted(set(drop_features) - set(FEATURES))
    if unknown_log:
        raise ValueError(f"log1p is only for numeric features; got {unknown_log}")
    if unknown_drop:
        raise ValueError(f"Unknown features to drop: {unknown_drop}")
    return log_features, drop_features


def make_preprocessor(log_features=(), drop_features=()) -> ColumnTransformer:
    """Build the MLP transformer; all learned statistics come from training data.

    With the default arguments the transformer is exactly the baseline
    (one-hot + standardization). log_features get log1p before standardization
    (E0); drop_features are left out of the input altogether (E6).
    """
    log_features, drop_features = _check_options(log_features, drop_features)
    categorical = [f for f in CATEGORICAL_FEATURES if f not in drop_features]
    numeric = [f for f in NUMERIC_FEATURES if f not in drop_features and f not in log_features]
    logged = [f for f in NUMERIC_FEATURES if f not in drop_features and f in log_features]
    binary = [f for f in BINARY_FEATURES if f not in drop_features]

    transformers = [
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            categorical,
        ),
        ("numeric", StandardScaler(), numeric),
    ]
    if logged:
        log_then_scale = Pipeline([
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric_log1p", log_then_scale, logged))
    transformers.append(("binary", "passthrough", binary))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def prepare_data(
    data_path: str | Path = DEFAULT_DATA_PATH,
    log_features=(),
    drop_features=(),
) -> DataSplits:
    """Create a stratified test split and a separate validation split.

    The 80% training partition is divided 75/25 into train and validation,
    yielding 60/20/20 overall. The default Project 2 CSV is a separate copy
    with exact duplicate rows removed; Project 1's original data stays intact.
    log_features and drop_features change only the input coding (E0, E6);
    the split itself is the same for every option.
    """
    log_features, drop_features = _check_options(log_features, drop_features)
    data = load_data(data_path)
    audit = audit_data(data)

    X = data[FEATURES]
    y = data[TARGET].astype(np.int64)

    X_train_validation, X_test, y_train_validation, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=0.25,
        stratify=y_train_validation,
        random_state=RANDOM_STATE,
    )

    preprocessor = make_preprocessor(log_features, drop_features)
    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32, copy=False)
    X_validation_processed = preprocessor.transform(X_validation).astype(np.float32, copy=False)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32, copy=False)

    partitions = [set(X_train.index), set(X_validation.index), set(X_test.index)]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Data split indices overlap.")

    for name, values in (
        ("train", X_train_processed),
        ("validation", X_validation_processed),
        ("test", X_test_processed),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"Processed {name} data contains NaN or infinite values.")

    # Track identical full rows crossing the train/test boundary for reporting.
    train_rows = X_train.copy()
    train_rows[TARGET] = y_train
    train_keys = set(map(tuple, train_rows.to_numpy().tolist()))
    test_rows = X_test.copy()
    test_rows[TARGET] = y_test
    audit["test_rows_identical_to_training_rows"] = sum(
        tuple(row) in train_keys for row in test_rows.to_numpy().tolist()
    )

    return DataSplits(
        X_train=X_train_processed,
        y_train=y_train.to_numpy(dtype=np.int64),
        X_validation=X_validation_processed,
        y_validation=y_validation.to_numpy(dtype=np.int64),
        X_test=X_test_processed,
        y_test=y_test.to_numpy(dtype=np.int64),
        train_indices=X_train.index.copy(),
        validation_indices=X_validation.index.copy(),
        test_indices=X_test.index.copy(),
        feature_names=preprocessor.get_feature_names_out(),
        preprocessor=preprocessor,
        audit=audit,
        options={"log_features": log_features, "drop_features": drop_features},
    )


def prepare_final_data(
    splits: DataSplits,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> FinalData:
    """Refit preprocessing on all 80% training rows after choices are frozen.

    Use only after architecture, hyperparameters, and training duration have
    been selected with the train/validation partitions. The test partition is
    transformed with this newly fitted preprocessor and remains held out.
    The input coding (log1p, dropped features) is taken from `splits`, so
    pass the splits made with the options selected in E0.
    """
    data = load_data(data_path)
    train_indices = splits.train_indices.append(splits.validation_indices)
    X_train = data.loc[train_indices, FEATURES]
    y_train = data.loc[train_indices, TARGET].astype(np.int64)
    X_test = data.loc[splits.test_indices, FEATURES]
    y_test = data.loc[splits.test_indices, TARGET].astype(np.int64)

    preprocessor = make_preprocessor(
        splits.options.get("log_features", ()),
        splits.options.get("drop_features", ()),
    )
    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32, copy=False)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32, copy=False)

    if not np.isfinite(X_train_processed).all() or not np.isfinite(X_test_processed).all():
        raise ValueError("Final processed data contains NaN or infinite values.")

    return FinalData(
        X_train=X_train_processed,
        y_train=y_train.to_numpy(dtype=np.int64),
        X_test=X_test_processed,
        y_test=y_test.to_numpy(dtype=np.int64),
        train_indices=train_indices,
        test_indices=splits.test_indices.copy(),
        feature_names=preprocessor.get_feature_names_out(),
        preprocessor=preprocessor,
    )
