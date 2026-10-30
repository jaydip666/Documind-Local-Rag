"""Unit tests for src.chunking — no external services required."""

import pytest

from src.chunking import Chunk, chunk_documents, chunk_text
from src.document_loader import LoadedPage


def test_chunk_text_splits_into_expected_word_counts() -> None:
    """Chunks should contain chunk_size words each, except possibly the last."""
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_text(text, chunk_size=4, overlap=0)

    assert chunks == [
        "word0 word1 word2 word3",
        "word4 word5 word6 word7",
        "word8 word9",
    ]


def test_chunk_text_applies_overlap() -> None:
    """Consecutive chunks should share `overlap` words at the boundary."""
    text = " ".join(f"w{i}" for i in range(6))
    chunks = chunk_text(text, chunk_size=4, overlap=2)

    assert chunks[0] == "w0 w1 w2 w3"
    assert chunks[1] == "w2 w3 w4 w5"


def test_chunk_text_empty_input_returns_empty_list() -> None:
    """Empty or whitespace-only text should yield no chunks."""
    assert chunk_text("   ", chunk_size=10, overlap=2) == []
    assert chunk_text("", chunk_size=10, overlap=2) == []


def test_chunk_text_invalid_overlap_raises() -> None:
    """overlap must be smaller than chunk_size and non-negative."""
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=5, overlap=5)
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=5, overlap=-1)
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=0, overlap=0)


def test_chunk_documents_preserves_source_metadata() -> None:
    """Each produced Chunk should carry its originating source and page."""
    pages = [
        LoadedPage(text=" ".join(f"w{i}" for i in range(5)), source="a.txt", page=None),
        LoadedPage(text=" ".join(f"w{i}" for i in range(5)), source="b.pdf", page=2),
    ]
    chunks = chunk_documents(pages, chunk_size=5, overlap=0)

    assert len(chunks) == 2
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert chunks[0].source == "a.txt" and chunks[0].page is None
    assert chunks[1].source == "b.pdf" and chunks[1].page == 2
    assert chunks[0].chunk_index == 0 and chunks[1].chunk_index == 0
