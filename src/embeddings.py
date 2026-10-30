"""Wraps sentence-transformers model loading and encoding."""

import functools

import numpy as np
from sentence_transformers import SentenceTransformer

from src.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModelError(Exception):
    """Raised when the embedding model fails to load or encode text."""


class EmbeddingModel:
    """Loads a sentence-transformers model and encodes text into vectors."""

    def __init__(self, model_name: str) -> None:
        """Load the sentence-transformers model.

        Args:
            model_name: Name of the sentence-transformers model to load
                (e.g. "all-MiniLM-L6-v2").

        Raises:
            EmbeddingModelError: If the model cannot be loaded (e.g. no network
                access and no local cache).
        """
        self.model_name = model_name
        try:
            logger.info("Loading embedding model '%s'", model_name)
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingModelError(
                f"Failed to load embedding model '{model_name}'. "
                f"Ensure it has been downloaded/cached locally: {exc}"
            ) from exc

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into embedding vectors.

        Args:
            texts: List of input strings to embed.

        Returns:
            A 2D numpy array of shape (len(texts), embedding_dim).

        Raises:
            EmbeddingModelError: If encoding fails.
        """
        if not texts:
            return np.empty((0, self.get_embedding_dim()))
        try:
            return self._model.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            )
        except Exception as exc:
            raise EmbeddingModelError(f"Failed to encode text batch: {exc}") from exc

    def get_embedding_dim(self) -> int:
        """Return the output vector dimension of the loaded model.

        Args:
            None.

        Returns:
            The embedding dimension as an integer.
        """
        return int(self._model.get_sentence_embedding_dimension())


@functools.lru_cache(maxsize=None)
def get_embedding_model(model_name: str) -> EmbeddingModel:
    """Return a process-wide cached EmbeddingModel for a given model name.

    This is a framework-agnostic singleton: it loads the underlying
    sentence-transformers model at most once per model_name per process, no
    matter how many times or from where it's called (Streamlit reruns,
    scripts, tests), avoiding repeated Hugging Face downloads/lookups on
    every call.

    Args:
        model_name: Name of the sentence-transformers model to load.

    Returns:
        A shared EmbeddingModel instance for this model_name.

    Raises:
        EmbeddingModelError: If the model cannot be loaded.
    """
    return EmbeddingModel(model_name)
