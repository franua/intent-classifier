import pytest
from server import app
from server import app_config
from server import model
import random
import string


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_inference_classes_are_loaded():
    assert app_config.get_inference_classes()


def test_ready_endpoint_returns_503_when_model_is_not_ready(client):
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.data == b"Not ready"


def test_ready_endpoint_returns_200_when_model_is_ready(client):
    model.load()
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.data == b"OK"


def test_intent_endpoint_with_valid_request(client):
    response = client.post(
        "/intent",
        json={
            "text": "what is the arrival time in san francisco for the 755 am flight leaving washington"
        },
    )
    assert response.status_code == 200
    assert "intents" in response.json

    intents = response.json["intents"]
    expected_intents = ["flight time", "flight number", "flight"]
    assert expected_intents == [v["label"] for v in intents]

    for intent in intents:
        assert isinstance(intent["score"], float) and intent["score"] > 0.0


@pytest.fixture
def response_400_fixtures():
    return [
        {
            "body": {
                "text": " ".join(
                    [
                        " ".join(
                            random.choices(string.ascii_letters, k=random.randint(2, 5))
                        )
                        for _ in range(61)
                    ]
                )
            },
            "response": {
                "label": "TEXT_TOO_LONG",
                "message": '"text" is too long.',
            },
        },
        {
            "body": {"text": "a" * 311},
            "response": {
                "label": "TEXT_TOO_LONG",
                "message": '"text" is too long.',
            },
        },
        {
            "body": {"asd": ""},
            "response": {
                "label": "TEXT_MISSING",
                "message": '"text" missing from request body.',
            },
        },
        {
            "body": {"text": ""},
            "response": {
                "label": "TEXT_MISSING",
                "message": '"text" missing from request body.',
            },
        },
        {
            "body": {},
            "response": {
                "label": "BODY_MISSING",
                "message": "Request doesn't have a body.",
            },
        },
    ]


def test_intent_endpoint_400_text_too_long(client, response_400_fixtures):
    for fixture in response_400_fixtures:
        response = client.post("/intent", json=fixture["body"])
        assert response.status_code == 400
        expected_response_body = fixture["response"]
        assert response.json == expected_response_body


def test_intent_endpoint_500(client):
    model.unload()
    response = client.post(
        "/intent",
        json={
            "text": "what is the arrival time in san francisco for the 755 am flight leaving washington"
        },
    )
    assert response.status_code == 500
    assert response.json == {
        "label": "INTERNAL_ERROR",
        "message": "Attempt to infer using a ZERO-SHOT pipeline when '<class 'NoneType'>' is loaded.",
    }
