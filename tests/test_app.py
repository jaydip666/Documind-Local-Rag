"""Unit tests for app.answer_question's retrieval-and-generation orchestration.

All external dependencies (embedder, vector store, LLM) are mocked.
"""

import dataclasses
from unittest.mock import MagicMock, patch

import app
from config import SETTINGS


def _make_chunk(score: float) -> dict:
    """Build a minimal retrieved-chunk dict with the given similarity score."""
    return {"text": "some text", "source": "doc.txt", "page": None, "score": score}


def test_answer_question_skips_llm_when_best_score_below_threshold() -> None:
    """A low best-match score should short-circuit to the refusal without calling the LLM."""
    mock_store = MagicMock()
    mock_store.search.return_value = [_make_chunk(0.2)]
    mock_llm = MagicMock()

    with patch.object(app, "get_embedder", return_value=MagicMock()), patch.object(
        app, "get_vector_store", return_value=mock_store
    ), patch.object(app, "get_llm_client", return_value=mock_llm), patch.object(
        app, "SETTINGS", dataclasses.replace(SETTINGS, min_relevance_score=0.5)
    ):
        answer, chunks = app.answer_question("totally unrelated question")

    assert answer == "I don't know based on the provided documents."
    assert chunks == []
    mock_llm.generate.assert_not_called()


def test_answer_question_runs_llm_when_best_score_meets_threshold() -> None:
    """A strong best-match score should proceed through the normal RAG pipeline."""
    high_score_chunks = [_make_chunk(0.9)]
    mock_store = MagicMock()
    mock_store.search.return_value = high_score_chunks
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "The answer is 42."

    with patch.object(app, "get_embedder", return_value=MagicMock()), patch.object(
        app, "get_vector_store", return_value=mock_store
    ), patch.object(app, "get_llm_client", return_value=mock_llm), patch.object(
        app, "SETTINGS", dataclasses.replace(SETTINGS, min_relevance_score=0.5)
    ):
        answer, chunks = app.answer_question("a well-matched question")

    assert answer == "The answer is 42."
    assert chunks == high_score_chunks
    mock_llm.generate.assert_called_once()


def test_answer_question_handles_no_chunks_without_threshold_check() -> None:
    """An empty retrieval result should still short-circuit before the LLM call."""
    mock_store = MagicMock()
    mock_store.search.return_value = []
    mock_llm = MagicMock()

    with patch.object(app, "get_embedder", return_value=MagicMock()), patch.object(
        app, "get_vector_store", return_value=mock_store
    ), patch.object(app, "get_llm_client", return_value=mock_llm):
        answer, chunks = app.answer_question("anything")

    assert "I don't know based on the provided documents" in answer
    assert chunks == []
    mock_llm.generate.assert_not_called()
