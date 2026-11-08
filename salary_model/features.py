"""Input columns and cleaning rules shared by training and prediction."""

import math

import pandas as pd
from word2number import w2n

EXPERIENCE = "experience"
SCORES = ["test_score", "interview_score"]
FEATURES = [EXPERIENCE, *SCORES]
TARGET = "salary"


def parse_experience(value):
    """Convert an experience value such as "four", "4" or 4 to years.

    A blank value means the candidate has no recorded experience, so it becomes 0.
    """
    if value is None or pd.isna(value):
        return 0.0

    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return 0.0
        try:
            years = float(text)
        except ValueError:
            try:
                years = w2n.word_to_num(text)
            except ValueError as error:
                raise ValueError(f"Could not read experience value: {value!r}") from error
    else:
        years = float(value)

    if not math.isfinite(years) or years < 0:
        raise ValueError(f"Could not read experience value: {value!r}")
    return float(years)


def parse_experience_column(frame):
    """Apply parse_experience to the single-column frame given by the pipeline."""
    return frame.iloc[:, 0].map(parse_experience).astype(float).to_frame()


def require_columns(frame, columns):
    """Raise a clear error if any expected column is missing."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
