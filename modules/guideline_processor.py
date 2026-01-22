"""Guideline processing utilities.

This module is responsible for extracting text from the local Uganda HIV guidelines PDF
and chunking it into retrievable segments with page metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from pypdf import PdfReader


@dataclass(frozen=True)
class GuidelineChunk:
    """A chunk of guideline text with traceability metadata."""

    chunk_id: str
    text: str
    source_path: str
    page_number: int


def _normalize_whitespace(text: str) -> str:
    # Rationale: PDF text extraction often contains irregular spacing/newlines. Normalizing
    # reduces noise for both embeddings and LLM explanations.
    return " ".join(text.replace("\u00a0", " ").split())


def _split_into_paragraphs(text: str) -> List[str]:
    # Rationale: paragraph-like splits produce more coherent retrieval units than fixed
    # character windows, while keeping implementation lightweight.
    raw_parts = [p.strip() for p in text.split("\n")]
    return [p for p in raw_parts if p]


def chunk_page_text(
    page_text: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> List[str]:
    """Chunk a single page of extracted PDF text.

    Notes:
    - This intentionally uses a simple heuristic chunker for demo reliability.
    - Chunking is performed within a page to preserve page-level traceability.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    normalized = _normalize_whitespace(page_text)
    if not normalized:
        return []

    paragraphs = _split_into_paragraphs(page_text)
    if not paragraphs:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_norm = _normalize_whitespace(para)
        if not para_norm:
            continue

        # If a single paragraph is very long, fall back to fixed windows.
        if len(para_norm) > chunk_size:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0

            start = 0
            while start < len(para_norm):
                end = min(start + chunk_size, len(para_norm))
                chunks.append(para_norm[start:end].strip())
                if end == len(para_norm):
                    break
                start = max(end - overlap, 0)
            continue

        if current_len + len(para_norm) + 1 > chunk_size and current:
            chunks.append(" ".join(current).strip())

            # Overlap: re-seed next chunk with the tail of the previous chunk.
            if overlap > 0:
                prev = chunks[-1]
                tail = prev[-overlap:]
                current = [tail]
                current_len = len(tail)
            else:
                current = []
                current_len = 0

        current.append(para_norm)
        current_len += len(para_norm) + 1

    if current:
        chunks.append(" ".join(current).strip())

    # Final cleanup pass.
    return [_normalize_whitespace(c) for c in chunks if _normalize_whitespace(c)]


def extract_page_texts(
    pdf_path: Path,
    *,
    max_pages: Optional[int] = None,
) -> List[str]:
    """Extract text from each page of a PDF.

    max_pages:
        Optional cap used to speed up smoke tests and local iteration.
    """

    reader = PdfReader(str(pdf_path))
    page_texts: List[str] = []

    total_pages = len(reader.pages)
    limit = total_pages if max_pages is None else min(max_pages, total_pages)

    for i in range(limit):
        page = reader.pages[i]
        text = page.extract_text() or ""
        page_texts.append(text)

    return page_texts


def process_guidelines(
    pdf_path: Path,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
    max_pages: Optional[int] = None,
) -> List[GuidelineChunk]:
    """Process a guideline PDF into chunks with metadata."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"Guideline PDF not found: {pdf_path}")

    page_texts = extract_page_texts(pdf_path, max_pages=max_pages)

    chunks: List[GuidelineChunk] = []
    for idx, page_text in enumerate(page_texts, start=1):
        for j, chunk_text in enumerate(
            chunk_page_text(page_text, chunk_size=chunk_size, overlap=overlap), start=1
        ):
            chunks.append(
                GuidelineChunk(
                    chunk_id=f"p{idx}_c{j}",
                    text=chunk_text,
                    source_path=str(pdf_path),
                    page_number=idx,
                )
            )

    return chunks


def iter_chunk_texts(chunks: Iterable[GuidelineChunk]) -> Iterable[str]:
    """Convenience iterator for embedding generation."""

    for c in chunks:
        yield c.text
