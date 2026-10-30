"""Wraps Ollama's REST API (localhost:11434) for local LLM answer generation.

This module never touches OpenSearch or the embedding model — it only knows
how to send a prompt to Ollama and return/stream the generated text.
"""

import json
from collections.abc import Iterator

import requests

from src.logger import get_logger

logger = get_logger(__name__)


class LLMClientError(Exception):
    """Raised when the Ollama server cannot be reached or returns an error."""


class LLMClient:
    """Thin client for Ollama's /api/generate endpoint."""

    def __init__(self, base_url: str, model: str, timeout: int) -> None:
        """Initialize the Ollama client.

        Args:
            base_url: Base URL of the Ollama server, e.g. "http://localhost:11434".
            model: Ollama model tag to use, e.g. "llama3.1".
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Stream a generated answer token-by-token from Ollama.

        Args:
            prompt: The full prompt to send to the model.

        Yields:
            Successive text fragments as they are generated.

        Raises:
            LLMClientError: If the Ollama server is unreachable, the model is
                missing, or the response cannot be parsed.
        """
        url = f"{self.base_url}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "stream": True}

        try:
            with requests.post(
                url, json=payload, stream=True, timeout=self.timeout
            ) as response:
                if response.status_code == 404:
                    raise LLMClientError(
                        f"Model '{self.model}' was not found on the Ollama server. "
                        f"Run: ollama pull {self.model}"
                    )
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        payload_line = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LLMClientError(
                            f"Received malformed response from Ollama: {exc}"
                        ) from exc

                    if "error" in payload_line:
                        raise LLMClientError(
                            f"Ollama returned an error: {payload_line['error']}"
                        )

                    fragment = payload_line.get("response", "")
                    if fragment:
                        yield fragment
                    if payload_line.get("done"):
                        break

        except requests.exceptions.ConnectionError as exc:
            raise LLMClientError(
                f"Could not connect to Ollama at {self.base_url}. "
                f"Is Ollama running (`ollama serve`)? {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMClientError(
                f"Ollama request timed out after {self.timeout}s. "
                f"Try a smaller model or increase OLLAMA_TIMEOUT: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMClientError(f"Request to Ollama failed: {exc}") from exc

    def generate(self, prompt: str) -> str:
        """Generate a full answer (non-streaming) from Ollama.

        Args:
            prompt: The full prompt to send to the model.

        Returns:
            The complete generated answer text.

        Raises:
            LLMClientError: If the Ollama server is unreachable or errors out.
        """
        return "".join(self.generate_stream(prompt))
