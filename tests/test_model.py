import json

import joblib
import numpy as np
import pandas as pd
import pytest

from salary_model.features import FEATURES, TARGET
from salary_model.predict import predict_file, predict_salaries
from salary_model.train import (
    DEFAULT_DATA,
    DEFAULT_MODEL,
    build_pipeline,
    cross_validated_scores,
    load_training_data,
    train,
)


@pytest.fixture(scope="module")
def data():
    return load_training_data(DEFAULT_DATA)


@pytest.fixture(scope="module")
def model(data):
    return build_pipeline().fit(data[FEATURES], data[TARGET])


def candidates(rows):
    return pd.DataFrame(rows, columns=FEATURES)


def test_model_matches_original_notebook_results(data, model):
    assert model.score(data[FEATURES], data[TARGET]) == pytest.approx(0.93052, abs=1e-4)
    prediction = model.predict(candidates([["four", 7, 8]]))[0]
    assert prediction == pytest.approx(61117.19, abs=0.01)


def test_missing_score_uses_training_median_not_the_new_files_median(model):
    """Regression test: the client file used to be filled with its own median."""
    low_scoring_batch = candidates([[None, np.nan, 7], ["two", 1, 5], ["two", 1, 5]])
    expected = model.predict(candidates([[None, 8, 7]]))[0]
    assert model.predict(low_scoring_batch)[0] == pytest.approx(expected)


def test_prediction_for_one_row_does_not_depend_on_other_rows(model):
    alone = model.predict(candidates([["ten", np.nan, 9]]))[0]
    together = model.predict(candidates([["ten", np.nan, 9], ["one", 2, 2]]))[0]
    assert alone == pytest.approx(together)


def test_saved_model_matches_a_fresh_fit(data, model):
    saved = joblib.load(DEFAULT_MODEL)
    assert saved.predict(data[FEATURES]) == pytest.approx(model.predict(data[FEATURES]))


def test_cross_validation_is_harder_than_training_data(data, model):
    scores = cross_validated_scores(data, repeats=5)
    assert scores["r2"] < model.score(data[FEATURES], data[TARGET])
    assert scores["mae"] > 0


def test_predict_salaries_adds_a_column_and_keeps_the_input(model):
    frame = candidates([["five", 5, 6]])
    result = predict_salaries(frame, model)
    assert list(result.columns) == [*FEATURES, "predicted_salary"]
    assert result.loc[0, "experience"] == "five"
    assert "predicted_salary" not in frame.columns


def test_predict_salaries_reports_missing_columns(model):
    with pytest.raises(ValueError, match="interview_score"):
        predict_salaries(pd.DataFrame({"experience": ["one"], "test_score": [5]}), model)


def test_invalid_experience_is_rejected(model):
    with pytest.raises(ValueError, match="banana"):
        model.predict(candidates([["banana", 5, 5]]))


def test_train_and_predict_write_their_files(tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    train(DEFAULT_DATA, model_path, metrics_path)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["rows"] == 51
    assert metrics["cross_validated"]["r2"] > metrics["average_salary_baseline"]["r2"]

    input_path = tmp_path / "input.csv"
    input_path.write_text("experience,test_score,interview_score\nfive,,6\n", encoding="utf-8")
    output_path = tmp_path / "output.csv"
    predict_file(input_path, output_path, model_path)
    assert pd.read_csv(output_path)["predicted_salary"].notnull().all()
