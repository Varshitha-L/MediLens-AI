import pandas as pd
import pytest
from src.risk_predictor import train_baseline, load_model, predict_one


def test_train_baseline_smoke(tmp_path):
    # Synthetic data only tests code execution; it is not a medical dataset.
    csv_path = tmp_path / "toy.csv"
    rows = []
    for i in range(80):
        rows.append({
            "age": 20 + i,
            "marker": float(i % 10),
            "group": "A" if i % 2 else "B",
            "target": "yes" if i % 3 else "no",
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    model_path = tmp_path / "models" / "model.joblib"
    result = train_baseline(csv_path, "target", model_path)
    assert model_path.exists()
    assert 0 <= result["metrics"]["f1_macro"] <= 1
    model = load_model(model_path)
    prediction = predict_one(model, {"age": 30, "marker": 2.0, "group": "A"})
    assert "prediction" in prediction
    assert "warning" in prediction


def test_missing_target_column_fails(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({"x": [1, 2], "label": ["a", "b"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Target column"):
        train_baseline(path, "missing", tmp_path / "model.joblib")

def test_prediction_rejects_missing_features(tmp_path):
    import pytest
    from src.risk_predictor import (
        train_baseline,
        predict_one,
        load_model,
    )

    path = tmp_path / "data.csv"
    pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55],
        "label": ["a", "b", "a", "b", "a", "b", "a", "b"],
    }).to_csv(path, index=False)

    model_path = tmp_path / "model.joblib"
    train_baseline(path, "label", model_path)
    model = load_model(model_path)

    with pytest.raises(
        ValueError,
        match="non-empty dictionary",
    ):
        predict_one(model, {})

def test_prediction_rejects_unexpected_features(tmp_path):
    import pytest
    from src.risk_predictor import train_baseline, predict_one, load_model

    path = tmp_path / "data.csv"
    pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55],
        "label": ["a", "b", "a", "b", "a", "b", "a", "b"],
    }).to_csv(path, index=False)

    model_path = tmp_path / "model.joblib"
    train_baseline(path, "label", model_path)
    model = load_model(model_path)

    with pytest.raises(ValueError, match="Unexpected features"):
        predict_one(model, {"age": 30, "unknown_column": 123})


def test_training_rejects_class_with_one_example(tmp_path):
    import pytest
    from src.risk_predictor import train_baseline

    path = tmp_path / "data.csv"
    pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45],
        "label": ["a", "a", "a", "a", "a", "b"],
    }).to_csv(path, index=False)

    with pytest.raises(ValueError, match="at least 2 examples"):
        train_baseline(path, "label", tmp_path / "model.joblib")
def test_prediction_rejects_missing_required_feature(tmp_path):
    import pytest
    from src.risk_predictor import (
        train_baseline,
        predict_one,
        load_model,
    )

    path = tmp_path / "data.csv"
    pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45, 50, 55],
        "label": ["a", "b", "a", "b", "a", "b", "a", "b"],
    }).to_csv(path, index=False)

    model_path = tmp_path / "model.joblib"
    train_baseline(path, "label", model_path)
    model = load_model(model_path)

    with pytest.raises(ValueError, match="Missing features"):
        predict_one(model, {"wrong_feature": 30})