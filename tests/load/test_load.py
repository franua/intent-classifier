import pytest
from server import app
from server import model
import time
import pandas as pd
import logging

sample_size = 300


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


@pytest.fixture
def request_fixtures():
    data = pd.concat(
        [
            pd.read_csv("data/atis/train.tsv", delimiter="\t", names=["text", "label"]),
            pd.read_csv("data/atis/test.tsv", delimiter="\t", names=["text", "label"]),
        ]
    )
    return data.sample(sample_size)[["text"]].to_dict(orient="records")


def test_performance(client, request_fixtures):
    model.load()
    start_time = time.time()
    for fixture in request_fixtures:
        client.post("/intent", json=fixture)

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(
        f"{len(request_fixtures)} samples inference: {elapsed_time:.2f} seconds"
    )

    ops = len(request_fixtures) / elapsed_time

    # arbitrary number, change to requirements
    assert ops > 5

    logging.info(f"The model's inference handles {ops} ops.")
