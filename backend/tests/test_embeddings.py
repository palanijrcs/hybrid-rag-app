"""Tests for the embedding model."""

import numpy as np
import pytest

from app.vectorstore.embeddings import get_embedding_model


@pytest.fixture(scope="module")
def model():
    return get_embedding_model()


def test_embedding_has_expected_shape(model):
    vectors = model.embed_texts(["hello world", "second text"])
    assert vectors.shape == (2, model.dimension)
    assert vectors.dtype == "float32"


def test_vectors_are_normalised(model):
    vector = model.embed_query("a pension for farmers")
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-4)


def test_similar_meaning_scores_higher_than_unrelated(model):
    question = model.embed_query("What pension do farmers receive?")
    related = model.embed_query(
        "A fixed pension of Rs.3,000 is provided to eligible small farmers."
    )
    unrelated = model.embed_query("The train leaves the station at midnight.")

    assert float(question @ related) > float(question @ unrelated)


def test_empty_input_returns_empty_array(model):
    assert model.embed_texts([]).shape == (0, model.dimension)