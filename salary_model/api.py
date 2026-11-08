"""Web app for the salary model.

Run from the repository root with: python -m salary_model.api
Then open http://127.0.0.1:8000 in a browser.
"""

import argparse
import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .features import FEATURES
from .predict import predict_salaries
from .train import DEFAULT_METRICS, DEFAULT_MODEL

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_EXPERIENCE = 60
MAX_SCORE = 10

LABELS = {
    "experience": "Experience",
    "test_score": "Test score",
    "interview_score": "Interview score",
}
FIELD_ERRORS = {
    "experience": f"Enter the years of experience, from 0 to {MAX_EXPERIENCE}.",
    "test_score": f"Enter a test score from 0 to {MAX_SCORE}.",
    "interview_score": f"Enter an interview score from 0 to {MAX_SCORE}.",
}


class Candidate(BaseModel):
    experience: float = Field(ge=0, le=MAX_EXPERIENCE)
    test_score: float = Field(ge=0, le=MAX_SCORE)
    interview_score: float = Field(ge=0, le=MAX_SCORE)


class Prediction(BaseModel):
    predicted_salary: int
    typical_error: int
    low: int
    high: int
    training_rows: int
    warnings: list[str]


@asynccontextmanager
async def lifespan(app):
    if not DEFAULT_MODEL.exists() or not DEFAULT_METRICS.exists():
        raise RuntimeError("Model files not found. Run: python -m salary_model.train")
    app.state.model = joblib.load(DEFAULT_MODEL)
    app.state.metrics = json.loads(DEFAULT_METRICS.read_text(encoding="utf-8"))
    yield


app = FastAPI(title="Salary Prediction", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError):
    """Return one readable message per field for the form to display."""
    errors = {}
    for problem in error.errors():
        location = problem["loc"]
        field = location[1] if len(location) > 1 else None
        if field in FIELD_ERRORS:
            errors[field] = FIELD_ERRORS[field]
        else:
            errors["form"] = "Send the candidate details as JSON."
    return JSONResponse(status_code=422, content={"errors": errors})


def range_warnings(candidate, ranges):
    """Explain which inputs fall outside the values the model was trained on."""
    warnings = []
    for name in FEATURES:
        value = getattr(candidate, name)
        low, high = ranges[name]["min"], ranges[name]["max"]
        if not low <= value <= high:
            warnings.append(
                f"{LABELS[name]} of {value:g} is outside the training data "
                f"({low:g} to {high:g}), so this estimate is less reliable."
            )
    return warnings


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/predict", response_model=Prediction)
def predict(candidate: Candidate, request: Request):
    model = request.app.state.model
    metrics = request.app.state.metrics

    frame = pd.DataFrame([candidate.model_dump()], columns=FEATURES)
    salary = int(predict_salaries(frame, model).loc[0, "predicted_salary"])
    typical_error = metrics["cross_validated"]["mae"]

    return Prediction(
        predicted_salary=salary,
        typical_error=typical_error,
        low=max(0, salary - typical_error),
        high=salary + typical_error,
        training_rows=metrics["rows"],
        warnings=range_warnings(candidate, metrics["input_ranges"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
