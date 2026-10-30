"""Load PDF/DOCX/TXT files into raw text plus source metadata.

This module only reads files from disk. It never chunks, embeds, or talks
to OpenSearch/Ollama — see chunking.py, embeddings.py, and vector_store.py.
"""

import os
from dataclasses import dataclass

from docx import Document as DocxDocument
from pypdf import PdfReader

from src.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@dataclass(frozen=True)
class LoadedPage:
    """One unit of raw extracted text with its source metadata.

    Attributes:
        text: The raw extracted text.
        source: The original filename the text was extracted from.
        page: 1-based page number when known (PDF), otherwise None.
    """

    text: str
    source: str
    page: int | None


class DocumentLoadError(Exception):
    """Raised when a source document cannot be read or parsed."""


def _load_pdf(file_path: str) -> list[LoadedPage]:
    """Extract text per page from a PDF file.

    Args:
        file_path: Absolute or relative path to the .pdf file.

    Returns:
        A list of LoadedPage, one per non-empty PDF page.

    Raises:
        DocumentLoadError: If the file cannot be opened or parsed.
    """
    source = os.path.basename(file_path)
    try:
        reader = PdfReader(file_path)
    except Exception as exc:
        raise DocumentLoadError(
            f"Could not open PDF '{source}'. It may be corrupted or encrypted: {exc}"
        ) from exc

    pages: list[LoadedPage] = []
    for index, pdf_page in enumerate(reader.pages, start=1):
        try:
            text = pdf_page.extract_text() or ""
        # pypdf raises no single specific error type for extraction failures.
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to extract text from page %d of '%s': %s", index, source, exc
            )
            continue
        if text.strip():
            pages.append(LoadedPage(text=text, source=source, page=index))

    if not pages:
        logger.error("No extractable text found in PDF '%s'", source)
    return pages


def _load_docx(file_path: str) -> list[LoadedPage]:
    """Extract text from a DOCX file as a single unit (DOCX has no fixed page concept).

    Args:
        file_path: Absolute or relative path to the .docx file.

    Returns:
        A list containing a single LoadedPage with the whole document text.

    Raises:
        DocumentLoadError: If the file cannot be opened or parsed.
    """
    source = os.path.basename(file_path)
    try:
        document = DocxDocument(file_path)
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    except Exception as exc:
        raise DocumentLoadError(
            f"Could not open DOCX '{source}'. It may be corrupted: {exc}"
        ) from exc

    if not text.strip():
        logger.error("No extractable text found in DOCX '%s'", source)
        return []
    return [LoadedPage(text=text, source=source, page=None)]


def _load_txt(file_path: str) -> list[LoadedPage]:
    """Read a plain text file as a single unit.

    Args:
        file_path: Absolute or relative path to the .txt file.

    Returns:
        A list containing a single LoadedPage with the file contents.

    Raises:
        DocumentLoadError: If the file cannot be read.
    """
    source = os.path.basename(file_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read()
    except OSError as exc:
        raise DocumentLoadError(f"Could not read TXT file '{source}': {exc}") from exc

    if not text.strip():
        logger.error("TXT file '%s' is empty", source)
        return []
    return [LoadedPage(text=text, source=source, page=None)]


def load_document(file_path: str) -> list[LoadedPage]:
    """Load a single PDF, DOCX, or TXT file into a list of LoadedPage objects.

    Args:
        file_path: Path to the source document.

    Returns:
        A list of LoadedPage objects extracted from the document.

    Raises:
        DocumentLoadError: If the extension is unsupported or the file cannot be parsed.
    """
    extension = os.path.splitext(file_path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file type '{extension}' for '{file_path}'. "
            f"Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    logger.info("Loading document '%s'", file_path)
    if extension == ".pdf":
        return _load_pdf(file_path)
    if extension == ".docx":
        return _load_docx(file_path)
    return _load_txt(file_path)


def load_documents_from_directory(directory: str) -> list[LoadedPage]:
    """Load every supported document found directly inside a directory.

    Args:
        directory: Path to the directory containing source documents.

    Returns:
        A combined list of LoadedPage objects from all supported files found.

    Raises:
        DocumentLoadError: If the directory does not exist.
    """
    if not os.path.isdir(directory):
        raise DocumentLoadError(f"Documents directory '{directory}' does not exist.")

    all_pages: list[LoadedPage] = []
    for filename in sorted(os.listdir(directory)):
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            continue
        if os.path.splitext(filename)[1].lower() not in SUPPORTED_EXTENSIONS:
            logger.debug("Skipping unsupported file '%s'", filename)
            continue
        try:
            all_pages.extend(load_document(file_path))
        except DocumentLoadError as exc:
            logger.error("Skipping '%s': %s", filename, exc)

    logger.info("Loaded %d page(s) from '%s'", len(all_pages), directory)
    return all_pages
