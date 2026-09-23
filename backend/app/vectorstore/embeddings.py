"""Turn text into embedding vectors."""

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wraps a sentence-transformers model behind a small, stable interface."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

        # The method was renamed in newer versions; support both.
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = self._model.get_sentence_embedding_dimension
        self.dimension: int = get_dimension()

        logger.info("Embedding model ready (dimension=%d)", self.dimension)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed many texts at once. Returns one row of numbers per text."""
        if not texts:
            return np.empty((0, self.dimension), dtype="float32")
        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,  # lets us compare with a simple dot product
            show_progress_bar=False,
        )
        return vectors.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single question."""
        return self.embed_texts([text])[0]


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    """Load the model once and reuse it."""
    return EmbeddingModel(get_settings().embedding_model)