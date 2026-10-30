"""Text splitting logic. chunk_size and overlap are always supplied by the caller
(read from config.py at the call site) rather than hardcoded here.
"""

from dataclasses import dataclass

from src.document_loader import LoadedPage
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Chunk:
    """A single chunk of text ready for embedding, with source metadata.

    Attributes:
        text: The chunk's text content.
        source: Original filename the chunk came from.
        page: 1-based page number when known, otherwise None.
        chunk_index: 0-based index of this chunk within its source page/document.
    """

    text: str
    source: str
    page: int | None
    chunk_index: int


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks.

    Args:
        text: The raw text to split.
        chunk_size: Maximum number of words per chunk. Must be positive.
        overlap: Number of words shared between consecutive chunks. Must be
            non-negative and smaller than chunk_size.

    Returns:
        A list of text chunks. Empty input yields an empty list.

    Raises:
        ValueError: If chunk_size is not positive or overlap is invalid.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
        start += step

    return chunks


def chunk_documents(
    pages: list[LoadedPage], chunk_size: int, overlap: int
) -> list[Chunk]:
    """Chunk a list of loaded document pages, preserving source metadata.

    Args:
        pages: Loaded pages/documents from document_loader.
        chunk_size: Maximum number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of Chunk objects across all input pages.
    """
    all_chunks: list[Chunk] = []
    for page in pages:
        pieces = chunk_text(page.text, chunk_size, overlap)
        for index, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(text=piece, source=page.source, page=page.page, chunk_index=index)
            )

    logger.info("Produced %d chunk(s) from %d page(s)", len(all_chunks), len(pages))
    return all_chunks
