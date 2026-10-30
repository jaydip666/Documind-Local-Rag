"""Streamlit entrypoint: chat-style UI for querying the local RAG assistant.

This module only wires the UI together — retrieval lives in vector_store.py,
prompt assembly in prompt_builder.py, and generation in llm_client.py.
"""

from typing import Any

import streamlit as st

from config import SETTINGS
from src.embeddings import EmbeddingModel, EmbeddingModelError, get_embedding_model
from src.llm_client import LLMClient, LLMClientError
from src.logger import get_logger
from src.prompt_builder import build_greeting_prompt, build_prompt, is_greeting
from src.vector_store import VectorStore, VectorStoreError

logger = get_logger(__name__)

st.set_page_config(page_title="Local RAG Assistant", page_icon="📚")


@st.cache_resource
def get_embedder() -> EmbeddingModel:
    """Load and cache the embedding model for the Streamlit session.

    Args:
        None.

    Returns:
        A ready-to-use EmbeddingModel instance.
    """
    return get_embedding_model(SETTINGS.embedding_model_name)


@st.cache_resource
def get_vector_store() -> VectorStore:
    """Create and cache the OpenSearch vector store client for the session.

    Args:
        None.

    Returns:
        A ready-to-use VectorStore instance.
    """
    return VectorStore(
        host=SETTINGS.opensearch_host,
        port=SETTINGS.opensearch_port,
        index_name=SETTINGS.opensearch_index_name,
        embedding_dim=SETTINGS.embedding_dim,
        use_ssl=SETTINGS.opensearch_use_ssl,
        verify_certs=SETTINGS.opensearch_verify_certs,
        username=SETTINGS.opensearch_username,
        password=SETTINGS.opensearch_password,
    )


def get_llm_client() -> LLMClient:
    """Build the Ollama LLM client for the session.

    Args:
        None.

    Returns:
        A ready-to-use LLMClient instance.
    """
    return LLMClient(
        base_url=SETTINGS.ollama_base_url,
        model=SETTINGS.ollama_model,
        timeout=SETTINGS.ollama_timeout,
    )


def answer_question(question: str) -> tuple[str, list[dict[str, Any]]]:
    """Run retrieval + generation for a single question.

    Args:
        question: The user's natural-language question.

    Returns:
        A tuple of (answer_text, retrieved_chunks). On any failure, answer_text
        is a friendly error message and retrieved_chunks is empty.
    """
    if is_greeting(question):
        try:
            llm = get_llm_client()
            answer = llm.generate(build_greeting_prompt(question))
        except LLMClientError as exc:
            logger.error("Generation failed: %s", exc)
            return (
                "Sorry, the LLM could not generate an answer. "
                + "Please check that Ollama is running with the configured model.",
                [],
            )
        return answer, []

    try:
        embedder = get_embedder()
        query_vector = embedder.encode([question])[0].tolist()
    except EmbeddingModelError as exc:
        logger.error("Embedding failed: %s", exc)
        return (
            "Sorry, the embedding model failed to process your question. Please try again.",
            [],
        )

    try:
        store = get_vector_store()
        chunks = store.search(query_vector, top_k=SETTINGS.top_k)
    except VectorStoreError as exc:
        logger.error("Retrieval failed: %s", exc)
        return (
            "Sorry, I couldn't reach the document index. "
            + "Please check that OpenSearch is running and try again.",
            [],
        )

    if not chunks:
        return (
            "I don't know based on the provided documents. "
            + "No relevant content was found — has `python ingest.py` been run?",
            [],
        )

    best_score = chunks[0].get("score", 0.0)
    if best_score < SETTINGS.min_relevance_score:
        logger.info(
            "Best match score %.3f is below threshold %.3f; skipping LLM call",
            best_score,
            SETTINGS.min_relevance_score,
        )
        return ("I don't know based on the provided documents.", [])

    prompt = build_prompt(question, chunks)

    try:
        llm = get_llm_client()
        answer = llm.generate(prompt)
    except LLMClientError as exc:
        logger.error("Generation failed: %s", exc)
        return (
            "Sorry, the local LLM could not generate an answer. "
            + "Please check that Ollama is running with the configured model.",
            chunks,
        )

    return answer, chunks


REFUSAL_PHRASES = (
    "i don't know",
    "i do not know",
    "cannot find",
    "couldn't reach",
    "could not generate",
    "failed to process",
)


def is_refusal(answer: str) -> bool:
    """Check whether an answer text is a refusal/error rather than a real answer.

    Args:
        answer: The assistant's answer text.

    Returns:
        True if the answer looks like a refusal or error message.
    """
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def render_sources(answer: str, chunks: list[dict[str, Any]]) -> None:
    """Render an expandable list of source citations under an answer.

    Args:
        answer: The assistant's answer text, used to detect refusals.
        chunks: The retrieved chunks used to build the answer.

    Returns:
        None.
    """
    if not chunks or is_refusal(answer):
        return
    with st.expander("Sources"):
        for chunk in chunks:
            label = chunk.get("source", "unknown")
            if chunk.get("page") is not None:
                label += f", page {chunk['page']}"
            st.markdown(f"**{label}**")
            st.caption(chunk.get("text", "")[:400])


def main() -> None:
    """Render the Streamlit chat UI and handle user interaction.

    Args:
        None.

    Returns:
        None.
    """
    st.title("DocuMind - Local RAG: Answers, grounded in your own documents.")
    st.caption(
        "Fully local & offline — your documents and questions never leave this machine."
    )

    if "history" not in st.session_state:
        st.session_state.history = []

    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message["content"], message.get("chunks", []))

    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, chunks = answer_question(question)
            st.markdown(answer)
            render_sources(answer, chunks)

        st.session_state.history.append(
            {"role": "assistant", "content": answer, "chunks": chunks}
        )


if __name__ == "__main__":
    main()
