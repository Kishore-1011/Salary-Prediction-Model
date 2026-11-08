"""Train the salary model, measure it on unseen rows and save the results.

Run from the repository root with: python -m salary_model.train
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .features import EXPERIENCE, FEATURES, SCORES, TARGET, parse_experience_column, require_columns

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "input_dataset.csv"
DEFAULT_MODEL = ROOT / "models" / "salary_model.joblib"
DEFAULT_METRICS = ROOT / "outputs" / "metrics.json"


def build_pipeline():
    """Create the model. Medians are learned in fit() and reused for every prediction."""
    preprocessing = ColumnTransformer(
        [
            ("experience", FunctionTransformer(parse_experience_column), [EXPERIENCE]),
            ("scores", SimpleImputer(strategy="median"), SCORES),
        ]
    )
    return Pipeline([("preprocessing", preprocessing), ("regression", LinearRegression())])


def load_training_data(path):
    frame = pd.read_csv(path)
    require_columns(frame, [*FEATURES, TARGET])
    return frame


def cross_validated_scores(frame, estimator=None, splits=5, repeats=50, seed=0):
    """Return the average R2 and absolute error on rows held out from training."""
    results = cross_validate(
        estimator if estimator is not None else build_pipeline(),
        frame[FEATURES],
        frame[TARGET],
        cv=RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=seed),
        scoring=("r2", "neg_mean_absolute_error"),
    )
    return {
        "r2": round(float(results["test_r2"].mean()), 3),
        "r2_std": round(float(results["test_r2"].std()), 3),
        "mae": round(float(-results["test_neg_mean_absolute_error"].mean())),
    }


def train(data_path=DEFAULT_DATA, model_path=DEFAULT_MODEL, metrics_path=DEFAULT_METRICS):
    """Fit on all rows, save the model and metrics, and return the metrics."""
    frame = load_training_data(data_path)
    model = build_pipeline().fit(frame[FEATURES], frame[TARGET])
    regression = model.named_steps["regression"]
    predictions = model.predict(frame[FEATURES])

    metrics = {
        "rows": len(frame),
        "duplicate_rows": int(frame.duplicated().sum()),
        "training_data": {
            "r2": round(float(model.score(frame[FEATURES], frame[TARGET])), 3),
            "mae": round(float(mean_absolute_error(frame[TARGET], predictions))),
        },
        "cross_validated": cross_validated_scores(frame),
        "cross_validated_without_duplicates": cross_validated_scores(frame.drop_duplicates()),
        "average_salary_baseline": cross_validated_scores(frame, estimator=DummyRegressor()),
        "intercept": round(float(regression.intercept_), 1),
        "salary_change_per_unit": {
            name: round(float(value), 1) for name, value in zip(FEATURES, regression.coef_)
        },
    }

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.model, args.metrics), indent=2))


if __name__ == "__main__":
    main()
