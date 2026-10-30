"""Central configuration module. All settings are loaded from environment variables (.env).

No other module should read os.environ directly or hardcode a default value —
everything flows through the constants defined here.
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    """Parse a boolean environment variable.

    Args:
        name: Environment variable name to read.
        default: Value to use when the variable is unset.

    Returns:
        The parsed boolean value.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    """Parse an integer environment variable.

    Args:
        name: Environment variable name to read.
        default: Value to use when the variable is unset or invalid.

    Returns:
        The parsed integer value.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    """Parse a float environment variable.

    Args:
        name: Environment variable name to read.
        default: Value to use when the variable is unset or invalid.

    Returns:
        The parsed float value.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Typed container for all application settings loaded from the environment."""

    documents_dir: str
    chunk_size: int
    chunk_overlap: int

    embedding_model_name: str
    embedding_dim: int

    opensearch_host: str
    opensearch_port: int
    opensearch_index_name: str
    opensearch_use_ssl: bool
    opensearch_verify_certs: bool
    opensearch_username: str
    opensearch_password: str

    top_k: int
    min_relevance_score: float

    ollama_base_url: str
    ollama_model: str
    ollama_timeout: int

    log_level: str


SETTINGS = Settings(
    documents_dir=os.getenv("DOCUMENTS_DIR", "documents"),
    chunk_size=_get_int("CHUNK_SIZE", 300),
    chunk_overlap=_get_int("CHUNK_OVERLAP", 50),
    embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
    embedding_dim=_get_int("EMBEDDING_DIM", 384),
    opensearch_host=os.getenv("OPENSEARCH_HOST", "localhost"),
    opensearch_port=_get_int("OPENSEARCH_PORT", 9200),
    opensearch_index_name=os.getenv("OPENSEARCH_INDEX_NAME", "rag_documents"),
    opensearch_use_ssl=_get_bool("OPENSEARCH_USE_SSL", False),
    opensearch_verify_certs=_get_bool("OPENSEARCH_VERIFY_CERTS", False),
    opensearch_username=os.getenv("OPENSEARCH_USERNAME", "admin"),
    opensearch_password=os.getenv("OPENSEARCH_PASSWORD", "admin"),
    top_k=_get_int("TOP_K", 4),
    min_relevance_score=_get_float("MIN_RELEVANCE_SCORE", 0.5),
    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1"),
    ollama_timeout=_get_int("OLLAMA_TIMEOUT", 120),
    log_level=os.getenv("LOG_LEVEL", "INFO"),
)
