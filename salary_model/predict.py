"""Predict salaries for a CSV of candidates.

Run from the repository root with: python -m salary_model.predict
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .features import FEATURES, require_columns
from .train import DEFAULT_MODEL, ROOT

DEFAULT_INPUT = ROOT / "data" / "hiring_test.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "output_file.csv"


def predict_salaries(frame, model):
    """Return a copy of frame with a predicted_salary column."""
    require_columns(frame, FEATURES)
    result = frame.copy()
    result["predicted_salary"] = model.predict(frame[FEATURES]).round().astype(int)
    return result


def predict_file(input_path=DEFAULT_INPUT, output_path=DEFAULT_OUTPUT, model_path=DEFAULT_MODEL):
    result = predict_salaries(pd.read_csv(input_path), joblib.load(model_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(predict_file(args.input, args.output, args.model).to_string(index=False))


if __name__ == "__main__":
    main()
