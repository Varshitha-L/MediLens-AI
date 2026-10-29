
"""Optional generic ML baseline trainer for a labeled tabular CSV.

Use only a dataset whose target and features are appropriate for the
specific research question. This module does not diagnose patients
and does not assume that a general ML dataset predicts medical risk.
"""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def train_baseline(
    csv_path: str | Path,
    target_column: str,
    model_path: str | Path = "models/risk_baseline.joblib",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, evaluate, and save a baseline classifier."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    data = pd.read_csv(csv_path)

    if data.empty:
        raise ValueError("Dataset is empty.")

    if target_column not in data.columns:
        raise ValueError(f"Target column {target_column!r} was not found.")

    data = data.dropna(subset=[target_column]).copy()

    if data.empty:
        raise ValueError("No rows remain after removing missing target values.")

    if data[target_column].nunique() < 2:
        raise ValueError("Target must contain at least two classes.")

    X = data.drop(columns=[target_column])
    y = data[target_column]

    if X.shape[1] == 0:
        raise ValueError("Dataset must contain at least one feature.")

    # Remove columns that contain no observed values.
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")

    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    classifier = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    # Validate that both partitions can contain every class.
    class_counts = y.value_counts()
    n_samples = len(y)
    n_test = ceil(test_size * n_samples)
    n_train = n_samples - n_test
    n_classes = y.nunique()

    if class_counts.min() < 2:
        raise ValueError(
            "Each target class needs at least 2 examples for a "
            "stratified train/test split. Add more examples per class."
        )

    if n_test < n_classes or n_train < n_classes:
        raise ValueError(
            "The dataset is too small for a stratified split with "
            "the selected test_size. Add more examples or choose "
            "a different test_size."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    # Use every target class, including classes absent from predictions.
    labels = pipeline.named_steps["classifier"].classes_.tolist()

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(
            precision_score(
                y_test, predictions, labels=labels,
                average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_test, predictions, labels=labels,
                average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y_test, predictions, labels=labels,
                average="macro", zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist(),
        "labels": [str(label) for label in labels],
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": list(X.columns),
        "target_column": target_column,
        "warning": (
            "Experimental baseline only. Metrics do not establish "
            "clinical safety, diagnostic validity, or suitability "
            "for patient-specific decisions."
        ),
    }

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)

    return {
        "model_path": str(destination),
        "metrics": metrics,
    }


def load_model(model_path: str | Path):
    """Load a previously saved pipeline; only load trusted model files."""

    path = Path(model_path)

    if not path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {path}")

    return joblib.load(path)


def predict_one(
    model: Any,
    features: dict[str, Any],
) -> dict[str, Any]:
    """Predict one row after validating its feature names."""

    if not isinstance(features, dict) or not features:
        raise ValueError("features must be a non-empty dictionary.")

    if not all(isinstance(name, str) for name in features):
        raise ValueError("All feature names must be strings.")

    named_steps = getattr(model, "named_steps", {})

    if "preprocessor" not in named_steps:
        raise ValueError(
            "Expected a fitted pipeline with a 'preprocessor' step."
        )

    preprocessor = named_steps["preprocessor"]

    if not hasattr(preprocessor, "feature_names_in_"):
        raise ValueError(
            "The preprocessor must be fitted before prediction."
        )

    expected_features = list(preprocessor.feature_names_in_)
    expected_set = set(expected_features)
    supplied_set = set(features)

    missing = sorted(expected_set - supplied_set)
    extra = sorted(supplied_set - expected_set)

    problems = []

    if missing:
        problems.append(f"Missing features: {missing}")

    if extra:
        problems.append(f"Unexpected features: {extra}")

    if problems:
        raise ValueError("; ".join(problems))

    # Preserve the feature order used during training.
    frame = pd.DataFrame([
        {name: features[name] for name in expected_features}
    ])

    prediction = model.predict(frame)[0]

    if hasattr(prediction, "item"):
        prediction = prediction.item()

    output = {"prediction": prediction}

    if hasattr(model, "predict_proba"):
        if "classifier" not in named_steps:
            raise ValueError(
                "Expected a pipeline step named 'classifier'."
            )

        probabilities = model.predict_proba(frame)[0]
        classes = named_steps["classifier"].classes_

        output["probabilities"] = {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities)
        }

    output["warning"] = (
        "Experimental model output; not a diagnosis or "
        "treatment recommendation."
    )

    return output
