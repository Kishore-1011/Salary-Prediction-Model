import numpy as np
import pandas as pd
import pytest

from salary_model.features import parse_experience, parse_experience_column, require_columns


@pytest.mark.parametrize(
    "value, expected",
    [
        ("four", 4.0),
        ("Eleven", 11.0),
        (" two ", 2.0),
        ("zero", 0.0),
        ("5", 5.0),
        ("2.5", 2.5),
        (7, 7.0),
        (None, 0.0),
        (np.nan, 0.0),
        ("", 0.0),
        ("   ", 0.0),
    ],
)
def test_parse_experience_reads_valid_values(value, expected):
    assert parse_experience(value) == expected


@pytest.mark.parametrize("value", ["banana", "-1", -3, "inf", "nan"])
def test_parse_experience_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="Could not read experience"):
        parse_experience(value)


def test_parse_experience_column_returns_one_numeric_column():
    frame = pd.DataFrame({"experience": ["two", None, "10"]})
    assert parse_experience_column(frame).iloc[:, 0].tolist() == [2.0, 0.0, 10.0]


def test_require_columns_names_what_is_missing():
    with pytest.raises(ValueError, match="test_score, salary"):
        require_columns(pd.DataFrame({"experience": []}), ["experience", "test_score", "salary"])
