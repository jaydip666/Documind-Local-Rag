"""Unit tests for src.embeddings using a mocked SentenceTransformer.

No real model download/load is performed.
"""

from unittest.mock import MagicMock, patch

from src.embeddings import get_embedding_model


def test_get_embedding_model_loads_once_per_model_name() -> None:
    """Repeated calls with the same model name must reuse the same instance
    and load the underlying SentenceTransformer only once."""
    get_embedding_model.cache_clear()
    with patch("src.embeddings.SentenceTransformer") as mock_st:
        mock_st.return_value = MagicMock()

        first = get_embedding_model("all-MiniLM-L6-v2")
        second = get_embedding_model("all-MiniLM-L6-v2")

        assert first is second
        mock_st.assert_called_once_with("all-MiniLM-L6-v2")


def test_get_embedding_model_loads_separately_per_model_name() -> None:
    """Different model names must each get their own loaded instance."""
    get_embedding_model.cache_clear()
    with patch("src.embeddings.SentenceTransformer") as mock_st:
        mock_st.return_value = MagicMock()

        first = get_embedding_model("all-MiniLM-L6-v2")
        second = get_embedding_model("all-mpnet-base-v2")

        assert first is not second
        assert mock_st.call_count == 2
