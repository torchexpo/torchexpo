"""BaseModel Tests"""
from unittest.mock import patch
import pytest
from requests.exceptions import HTTPError

from torchexpo.model import model, MODEL_QUERY


MODEL_DATA = {
    "data": {
        "model": {
            "slug": "test-publisher-slug",
            "name": "test-name",
            "task": {
                "slug": "image-classification",
                "name": "Image Classification"
            },
            "publisher": {
                "name": "test-publisher"
            },
            "dataset": {
                "name": "test-dataset"
            }
        }
    }
}


def test_model(requests_mock):
    requests_mock.get(
        url=f"https://torchexpo.fly.dev/graphql", json=MODEL_DATA)
    test_model = model(slug="test-publisher-slug", initialize=False)
    assert test_model.name == "test-name"
    assert test_model.task == "Image Classification"
    assert test_model.publisher == "test-publisher"
    assert test_model.dataset == "test-dataset"


def test_model_failed_exception(requests_mock):
    requests_mock.get(
        url=f"https://torchexpo.fly.dev/graphql", json={})
    with patch("requests.get", side_effect=HTTPError()):
        with pytest.raises(HTTPError):
            test_model = model(slug="test-publisher-slug", initialize=False)
            assert test_model.name == ""
            assert test_model.task == ""
            assert test_model.publisher == ""
            assert test_model.dataset == ""
