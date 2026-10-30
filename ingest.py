"""CLI script: load documents -> chunk -> embed -> index into OpenSearch.

Usage:
    python ingest.py

This script never talks to Ollama — it only prepares the searchable index
that app.py later queries.
"""

import sys

from config import SETTINGS
from src.chunking import chunk_documents
from src.document_loader import DocumentLoadError, load_documents_from_directory
from src.embeddings import EmbeddingModel, EmbeddingModelError
from src.logger import get_logger
from src.vector_store import VectorStore, VectorStoreError

logger = get_logger(__name__)


def run_ingestion() -> None:
    """Run the full ingestion pipeline: load, chunk, embed, and index documents.

    Args:
        None.

    Returns:
        None.
    """
    try:
        pages = load_documents_from_directory(SETTINGS.documents_dir)
    except DocumentLoadError as exc:
        logger.error("Ingestion aborted: %s", exc)
        sys.exit(1)

    if not pages:
        logger.error(
            "No readable documents found in '%s'. Add PDF/DOCX/TXT files and retry.",
            SETTINGS.documents_dir,
        )
        sys.exit(1)

    chunks = chunk_documents(pages, SETTINGS.chunk_size, SETTINGS.chunk_overlap)
    if not chunks:
        logger.error("No chunks were produced from the loaded documents.")
        sys.exit(1)

    try:
        embedder = EmbeddingModel(SETTINGS.embedding_model_name)
    except EmbeddingModelError as exc:
        logger.error("Ingestion aborted: %s", exc)
        sys.exit(1)

    texts = [chunk.text for chunk in chunks]
    try:
        vectors = embedder.encode(texts)
    except EmbeddingModelError as exc:
        logger.error("Ingestion aborted: %s", exc)
        sys.exit(1)

    documents = [
        {
            "text": chunk.text,
            "source": chunk.source,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
            "embedding": vectors[i].tolist(),
        }
        for i, chunk in enumerate(chunks)
    ]

    try:
        store = VectorStore(
            host=SETTINGS.opensearch_host,
            port=SETTINGS.opensearch_port,
            index_name=SETTINGS.opensearch_index_name,
            embedding_dim=SETTINGS.embedding_dim,
            use_ssl=SETTINGS.opensearch_use_ssl,
            verify_certs=SETTINGS.opensearch_verify_certs,
            username=SETTINGS.opensearch_username,
            password=SETTINGS.opensearch_password,
        )
        store.create_index()
        indexed_count = store.index_chunks(documents)
    except VectorStoreError as exc:
        logger.error("Ingestion aborted: %s", exc)
        sys.exit(1)

    logger.info(
        "Ingestion complete: %d chunk(s) from %d page(s) indexed into '%s'",
        indexed_count,
        len(pages),
        SETTINGS.opensearch_index_name,
    )


if __name__ == "__main__":
    run_ingestion()
