import pytest
from fastapi.testclient import TestClient

from salary_model.api import app

VALID = {"experience": 4, "test_score": 7, "interview_score": 8}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_home_page_shows_the_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="candidate-form"' in response.text


@pytest.mark.parametrize("asset", ["app.js", "style.css"])
def test_page_assets_are_served(client, asset):
    assert client.get(f"/static/{asset}").status_code == 200


def test_prediction_matches_the_model(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_salary"] == 61117
    assert body["low"] == body["predicted_salary"] - body["typical_error"]
    assert body["high"] == body["predicted_salary"] + body["typical_error"]
    assert body["training_rows"] == 51
    assert body["warnings"] == []


def test_inputs_at_the_edge_of_training_data_have_no_warning(client):
    response = client.post("/predict", json={"experience": 11, "test_score": 10, "interview_score": 10})
    assert response.json()["warnings"] == []


def test_inputs_outside_training_data_are_flagged(client):
    response = client.post("/predict", json={"experience": 15, "test_score": 2, "interview_score": 8})
    assert response.status_code == 200
    warnings = response.json()["warnings"]
    assert len(warnings) == 2
    assert "Experience of 15" in warnings[0]
    assert "Test score of 2" in warnings[1]


@pytest.mark.parametrize(
    "field, value",
    [
        ("experience", -1),
        ("experience", 61),
        ("experience", None),
        ("test_score", 10.5),
        ("test_score", "abc"),
        ("interview_score", -0.1),
        ("interview_score", None),
    ],
)
def test_invalid_field_gets_a_message_for_that_field(client, field, value):
    response = client.post("/predict", json={**VALID, field: value})
    assert response.status_code == 422
    errors = response.json()["errors"]
    assert list(errors) == [field]


def test_all_blank_fields_are_reported_together(client):
    response = client.post("/predict", json={"experience": None, "test_score": None, "interview_score": None})
    assert set(response.json()["errors"]) == {"experience", "test_score", "interview_score"}


def test_malformed_request_gets_a_readable_error(client):
    response = client.post("/predict", content="not json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert "form" in response.json()["errors"]
