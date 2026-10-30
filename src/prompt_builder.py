"""Builds the final LLM prompt from retrieved chunks and the user's question."""

import re
from typing import Any

from src.logger import get_logger

logger = get_logger(__name__)

SYSTEM_INSTRUCTIONS = (
    "You are a helpful assistant that answers questions using ONLY the "
    "context provided below. If the answer is not contained in the context, "
    'say "I don\'t know based on the provided documents." Do not use any '
    "outside knowledge, and do not make up information."
)

FRIENDLY_SYSTEM_INSTRUCTIONS = (
    "You are a friendly assistant. Respond naturally and briefly to this "
    "greeting or casual remark. Do not mention documents, context, or "
    "sources — this is just small talk."
)

# Multi-word conversational idioms are normalized to a single joined token
# before word-level classification, so phrases like "thank you" or "theek
# hai" are recognized as one unit instead of two separately-judged words.
_IDIOM_PATTERNS = (
    (re.compile(r"\bthank\s*you\b"), "thankyou"),
    (re.compile(r"\bno\s*problem\b"), "noproblem"),
    (re.compile(r"\bhow\s*are\s*you\b"), "howareyou"),
    (re.compile(r"\bwhat'?s\s*up\b"), "whatsup"),
    (re.compile(r"\bsee\s*you\b"), "seeyou"),
    (re.compile(r"\btake\s*care\b"), "takecare"),
    (re.compile(r"\bcatch\s*(you\s*)?later\b"), "catchlater"),
    (re.compile(r"\bgood\s*(morning|afternoon|evening|night)\b"), r"good\1"),
    (re.compile(r"\bkaise\s*ho\b"), "kaiseho"),
    (re.compile(r"\bkya\s*haal\s*(hai)?\b"), "kyahaal"),
    (re.compile(r"\bsab\s*theek\b"), "theekhai"),
    (re.compile(r"\btheek\s*(hai|ho)?\b"), "theekhai"),
    (re.compile(r"\bthik\s*(hai|ho)?\b"), "theekhai"),
)

# Bag-of-words vocabulary for conversational (non-informational) messages:
# greetings, farewells, thanks, acknowledgments, and casual small talk, in
# both English and common Hinglish forms. Deliberately excludes bare question
# words ("what", "how", "why" ...) so real questions never get swallowed here.
_CONVERSATIONAL_WORDS = {
    # greetings
    "hi", "hii", "hiii", "hello", "hey", "heyy",
    "there",
    # farewells
    "bye", "goodbye", "seeyou", "takecare", "catchlater",
    # gratitude
    "thanks", "thank", "thankyou",
    # acknowledgments / affirmations / reactions
    "ok", "okay", "sure", "alright", "fine", "cool", "nice",
    "great", "good", "awesome", "perfect", "got", "it", "noted", "understood",
    "welcome",
    # yes / no
    "yes", "yeah", "yep", "yup", "haan", "han",
    "no", "nope", "nah", "nahi", "nahin",
    # casual state-of-being idioms (joined by _IDIOM_PATTERNS above)
    "goodmorning", "goodafternoon", "goodevening", "goodnight",
    "howareyou", "whatsup", "kaiseho", "kyahaal", "theekhai", "noproblem",
}

_MAX_CONVERSATIONAL_WORDS = 6


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for classification.

    Args:
        text: Raw user message.

    Returns:
        A normalized string with only lowercase word characters and spaces.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_greeting(question: str) -> bool:
    """Classify a message as conversational (vs. an informational document query).

    Covers greetings, farewells, thanks, acknowledgments ("ok", "fine",
    "cool"), and casual small talk in English and Hinglish (e.g. "haan",
    "theek hai"). This is a bag-of-words rule: a short message is
    conversational only if *every* one of its words is in the conversational
    vocabulary. Any unrecognized word — a topic, name, or a question word
    like "what"/"how" used on its own — routes the message to the normal
    retrieval pipeline instead, so real document questions are never
    misclassified as small talk.

    Args:
        question: The user's raw message.

    Returns:
        True if the message is small talk rather than a factual question.
    """
    normalized = _normalize(question)
    if not normalized:
        return False

    for pattern, replacement in _IDIOM_PATTERNS:
        normalized = pattern.sub(replacement, normalized)

    words = normalized.split()
    if not words or len(words) > _MAX_CONVERSATIONAL_WORDS:
        return False

    return all(word in _CONVERSATIONAL_WORDS for word in words)


def build_greeting_prompt(question: str) -> str:
    """Build a lightweight prompt for greetings/small talk (no retrieval).

    Args:
        question: The user's greeting or casual message.

    Returns:
        A prompt instructing the LLM to reply naturally, without any
        document context.
    """
    return f"{FRIENDLY_SYSTEM_INSTRUCTIONS}\n\nMessage: {question}\nReply:"


def _format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered, source-labeled context block.

    Args:
        chunks: Retrieved chunk dicts with "text", "source", and "page" keys.

    Returns:
        A single string containing all chunks, each labeled with its source.
    """
    lines = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page")
        label = f"{source}" + (f", page {page}" if page is not None else "")
        lines.append(f"[{index}] (source: {label})\n{chunk.get('text', '')}")
    return "\n\n".join(lines)


def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Build the final prompt sent to the LLM.

    Args:
        question: The user's natural-language question.
        chunks: Retrieved context chunks, each a dict with "text", "source",
            and "page" keys.

    Returns:
        The complete prompt string combining instructions, context, and question.
    """
    if not chunks:
        logger.debug("Building prompt with no retrieved context")
        context_block = "(no relevant context found)"
    else:
        context_block = _format_context(chunks)

    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt
