"""OpenSearch client wrapper: index creation, bulk insert, and k-NN search.

This module owns all OpenSearch interaction. It never talks to Ollama or the
embedding model directly — callers pass in already-computed vectors.
"""

from typing import Any

from opensearchpy import OpenSearch
from opensearchpy.exceptions import NotFoundError, OpenSearchException
from opensearchpy.helpers import bulk

from src.logger import get_logger

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when an OpenSearch operation fails."""


class VectorStore:
    """Manages a k-NN enabled OpenSearch index of document chunks."""

    def __init__(
        self,
        host: str,
        port: int,
        index_name: str,
        embedding_dim: int,
        use_ssl: bool = False,
        verify_certs: bool = False,
        username: str | None = None,
        password: str | None = None,
        client: OpenSearch | None = None,
    ) -> None:
        """Initialize the vector store.

        Args:
            host: OpenSearch hostname.
            port: OpenSearch REST port.
            index_name: Name of the index to read/write.
            embedding_dim: Dimension of stored embedding vectors.
            use_ssl: Whether to connect over HTTPS.
            verify_certs: Whether to verify TLS certificates.
            username: Basic auth username, if required.
            password: Basic auth password, if required.
            client: Optional pre-built OpenSearch client, primarily used to
                inject a mock client in tests. When omitted, a real client is
                constructed from the other arguments.

        Raises:
            VectorStoreError: If a real client needs to be built and construction fails.
        """
        self.index_name = index_name
        self.embedding_dim = embedding_dim

        if client is not None:
            self._client = client
            return

        try:
            self._client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=(username, password) if username else None,
                use_ssl=use_ssl,
                verify_certs=verify_certs,
            )
        except Exception as exc:
            raise VectorStoreError(
                f"Failed to create OpenSearch client for {host}:{port}: {exc}"
            ) from exc

    def create_index(self) -> None:
        """Create the k-NN index if it does not already exist.

        Args:
            None.

        Returns:
            None.

        Raises:
            VectorStoreError: If index existence check or creation fails.
        """
        try:
            if self._client.indices.exists(index=self.index_name):
                logger.info("Index '%s' already exists", self.index_name)
                return
        except OpenSearchException as exc:
            raise VectorStoreError(
                f"Could not reach OpenSearch to check index '{self.index_name}'. "
                f"Is the cluster running and reachable? {exc}"
            ) from exc

        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": self.embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                        },
                    },
                }
            },
        }
        try:
            self._client.indices.create(index=self.index_name, body=body)
            logger.info("Created index '%s'", self.index_name)
        except OpenSearchException as exc:
            raise VectorStoreError(
                f"Failed to create index '{self.index_name}': {exc}"
            ) from exc

    def index_chunks(self, chunks: list[dict[str, Any]]) -> int:
        """Bulk insert chunk documents (each already carrying its embedding).

        Args:
            chunks: List of dicts each containing at least "text", "source",
                "page", "chunk_index", and "embedding" keys.

        Returns:
            The number of documents successfully indexed.

        Raises:
            VectorStoreError: If the bulk insert fails.
        """
        if not chunks:
            logger.info("No chunks to index")
            return 0

        actions = [{"_index": self.index_name, "_source": chunk} for chunk in chunks]
        try:
            success_count, errors = bulk(self._client, actions, raise_on_error=False)
        except OpenSearchException as exc:
            raise VectorStoreError(
                f"Bulk indexing to OpenSearch failed. Is the cluster reachable? {exc}"
            ) from exc

        if errors:
            logger.error("Encountered %d error(s) during bulk indexing", len(errors))
        logger.info("Indexed %d chunk(s) into '%s'", success_count, self.index_name)
        return success_count

    def search(self, query_vector: list[float], top_k: int) -> list[dict[str, Any]]:
        """Run a k-NN search and return the top matching chunks.

        Args:
            query_vector: Embedding vector for the user's question.
            top_k: Number of nearest chunks to retrieve.

        Returns:
            A list of dicts with "text", "source", "page", and "score" keys,
            ordered from most to least relevant. Returns an empty list if the
            index does not exist yet.

        Raises:
            VectorStoreError: If the search request fails for a reason other
                than a missing index.
        """
        body = {
            "size": top_k,
            "query": {"knn": {"embedding": {"vector": query_vector, "k": top_k}}},
        }
        try:
            response = self._client.search(index=self.index_name, body=body)
        except NotFoundError:
            logger.error(
                "Index '%s' does not exist. Run ingest.py first.", self.index_name
            )
            return []
        except OpenSearchException as exc:
            raise VectorStoreError(
                f"OpenSearch query failed. Is the cluster reachable? {exc}"
            ) from exc

        hits = response.get("hits", {}).get("hits", [])
        results = [
            {
                "text": hit["_source"].get("text", ""),
                "source": hit["_source"].get("source", "unknown"),
                "page": hit["_source"].get("page"),
                "score": hit.get("_score", 0.0),
            }
            for hit in hits
        ]
        logger.info("Retrieved %d chunk(s) for query", len(results))
        return results
