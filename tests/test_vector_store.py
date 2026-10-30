"""Unit tests for src.vector_store using a mocked OpenSearch client.

No live OpenSearch connection is required or made.
"""

from unittest.mock import MagicMock, patch

import pytest
from opensearchpy.exceptions import NotFoundError

from src.vector_store import VectorStore, VectorStoreError


@pytest.fixture
def mock_client() -> MagicMock:
    """Build a MagicMock standing in for an opensearchpy.OpenSearch client."""
    return MagicMock()


@pytest.fixture
def store(mock_client: MagicMock) -> VectorStore:
    """Build a VectorStore wired to the mocked OpenSearch client."""
    return VectorStore(
        host="localhost",
        port=9200,
        index_name="test_index",
        embedding_dim=384,
        client=mock_client,
    )


def test_create_index_skips_when_index_exists(
    store: VectorStore, mock_client: MagicMock
) -> None:
    """create_index should not attempt creation if the index already exists."""
    mock_client.indices.exists.return_value = True

    store.create_index()

    mock_client.indices.create.assert_not_called()


def test_create_index_creates_when_missing(
    store: VectorStore, mock_client: MagicMock
) -> None:
    """create_index should build a knn-enabled index when it does not exist."""
    mock_client.indices.exists.return_value = False

    store.create_index()

    mock_client.indices.create.assert_called_once()
    _, kwargs = mock_client.indices.create.call_args
    assert kwargs["index"] == "test_index"
    assert kwargs["body"]["settings"]["index"]["knn"] is True
    assert kwargs["body"]["mappings"]["properties"]["embedding"]["dimension"] == 384


def test_search_returns_formatted_hits(
    store: VectorStore, mock_client: MagicMock
) -> None:
    """search should map raw OpenSearch hits into simplified result dicts."""
    mock_client.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_source": {"text": "chunk one", "source": "doc.txt", "page": None},
                    "_score": 0.9,
                },
                {
                    "_source": {"text": "chunk two", "source": "doc.pdf", "page": 3},
                    "_score": 0.7,
                },
            ]
        }
    }

    results = store.search(query_vector=[0.1] * 384, top_k=2)

    assert len(results) == 2
    assert results[0] == {
        "text": "chunk one",
        "source": "doc.txt",
        "page": None,
        "score": 0.9,
    }
    assert results[1] == {
        "text": "chunk two",
        "source": "doc.pdf",
        "page": 3,
        "score": 0.7,
    }


def test_search_returns_empty_list_when_index_missing(
    store: VectorStore, mock_client: MagicMock
) -> None:
    """search should degrade gracefully to an empty list if the index is absent."""
    mock_client.search.side_effect = NotFoundError(404, "index_not_found_exception", {})

    results = store.search(query_vector=[0.1] * 384, top_k=2)

    assert results == []


def test_index_chunks_returns_count_from_bulk_helper(store: VectorStore) -> None:
    """index_chunks should delegate to opensearchpy.helpers.bulk and return its count."""
    chunks = [
        {
            "text": "a",
            "source": "a.txt",
            "page": None,
            "chunk_index": 0,
            "embedding": [0.1],
        },
    ]

    with patch("src.vector_store.bulk", return_value=(1, [])) as mock_bulk:
        count = store.index_chunks(chunks)

    assert count == 1
    mock_bulk.assert_called_once()


def test_index_chunks_empty_input_skips_bulk_call(store: VectorStore) -> None:
    """index_chunks should short-circuit and avoid calling bulk for empty input."""
    with patch("src.vector_store.bulk") as mock_bulk:
        count = store.index_chunks([])

    assert count == 0
    mock_bulk.assert_not_called()


def test_create_index_raises_vector_store_error_on_failure(
    store: VectorStore, mock_client: MagicMock
) -> None:
    """A connection failure while checking index existence should raise VectorStoreError."""
    from opensearchpy.exceptions import OpenSearchException

    mock_client.indices.exists.side_effect = OpenSearchException("boom")

    with pytest.raises(VectorStoreError):
        store.create_index()
