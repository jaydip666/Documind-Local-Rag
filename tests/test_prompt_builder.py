"""Unit tests for src.prompt_builder — no external services required."""

import pytest

from src.prompt_builder import build_prompt, is_greeting


def test_build_prompt_includes_question_and_context() -> None:
    """The prompt should include the question text and every chunk's text."""
    chunks = [
        {"text": "Paris is the capital of France.", "source": "geo.txt", "page": None},
        {
            "text": "It has a population of over 2 million.",
            "source": "geo.txt",
            "page": None,
        },
    ]
    prompt = build_prompt("What is the capital of France?", chunks)

    assert "What is the capital of France?" in prompt
    assert "Paris is the capital of France." in prompt
    assert "It has a population of over 2 million." in prompt
    assert "geo.txt" in prompt


def test_build_prompt_includes_page_numbers_when_present() -> None:
    """Chunks with a page number should show it in the formatted context."""
    chunks = [{"text": "Some fact.", "source": "manual.pdf", "page": 5}]
    prompt = build_prompt("A question?", chunks)

    assert "manual.pdf, page 5" in prompt


def test_build_prompt_handles_no_context() -> None:
    """With no retrieved chunks, the prompt should still be well-formed."""
    prompt = build_prompt("Anything?", [])

    assert "Anything?" in prompt
    assert "no relevant context found" in prompt


def test_build_prompt_instructs_no_hallucination() -> None:
    """The system instructions must forbid answering outside the given context."""
    prompt = build_prompt("Q?", [])

    assert "ONLY the" in prompt
    assert "I don't know based on the provided documents." in prompt


@pytest.mark.parametrize(
    "message",
    [
        "hi",
        "Hello!",
        "hey there",
        "fine",
        "fine, thanks",
        "thanks",
        "thank you",
        "ok",
        "okay cool",
        "good",
        "cool",
        "haan",
        "nahi",
        "theek hai",
        "sab theek",
        "bye",
        "goodbye, take care",
        "how are you",
        "what's up",
        "kaise ho",
        "good morning",
    ],
)
def test_is_greeting_detects_conversational_messages(message: str) -> None:
    """Greetings, acknowledgments, and casual small talk should be conversational."""
    assert is_greeting(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "What is the refund policy?",
        "How does the leave approval process work?",
        "good documentation practices for onboarding",
        "explain the pricing model",
        "no of employees mentioned in the report",
        "",
    ],
)
def test_is_greeting_ignores_informational_queries(message: str) -> None:
    """Real document questions must not be misclassified as small talk."""
    assert is_greeting(message) is False
