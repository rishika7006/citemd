"""Section-aware chunking.

The chunker keeps section and page metadata attached to every chunk so that retrieved
evidence can be cited precisely. It is pure Python with no heavy dependencies, which keeps
it fast to test.

Strategy:
  1. Split the document into sections on Markdown-style headings (``# ...``) when present,
     otherwise treat the whole document as one untitled section.
  2. Within each section, pack paragraphs greedily up to ``max_chars``, carrying a small
     character overlap between adjacent chunks so context is not lost at boundaries.
  3. A single paragraph longer than ``max_chars`` is hard-split on whitespace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from citemd.models import Chunk, Document

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$")
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


@dataclass
class _Section:
    title: str | None
    text: str


def _split_sections(text: str) -> list[_Section]:
    """Split text into sections on Markdown headings; fall back to a single section."""
    lines = text.splitlines()
    sections: list[_Section] = []
    current_title: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(_Section(title=current_title, text=body))

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            flush()
            buffer = []
            current_title = m.group(1).strip()
        else:
            buffer.append(line)
    flush()

    if not sections:
        body = text.strip()
        if body:
            sections.append(_Section(title=None, text=body))
    return sections


def _hard_split(paragraph: str, max_chars: int) -> list[str]:
    """Split an oversized paragraph on word boundaries into <= max_chars pieces."""
    words = paragraph.split()
    pieces: list[str] = []
    current = ""
    for word in words:
        if current and len(current) + 1 + len(word) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        pieces.append(current)
    return pieces


def _pack(paragraphs: list[str], max_chars: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into chunks with a trailing character overlap."""
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_hard_split(para, max_chars))
            continue
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > max_chars and current:
            chunks.append(current)
            tail = current[-overlap:] if overlap > 0 else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_text(
    text: str,
    *,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[tuple[str | None, str]]:
    """Chunk raw text, returning ``(section_title, chunk_text)`` pairs.

    Exposed separately from :func:`chunk_document` so it can be unit-tested without
    constructing a full Document.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and < max_chars")

    results: list[tuple[str | None, str]] = []
    for section in _split_sections(text):
        paragraphs = [p.strip() for p in _PARA_SPLIT_RE.split(section.text) if p.strip()]
        for chunk in _pack(paragraphs, max_chars, overlap):
            results.append((section.title, chunk))
    return results


def chunk_document(
    document: Document,
    *,
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    """Chunk a Document into Chunks, preserving source, section, and access metadata."""
    chunks: list[Chunk] = []
    for index, (section, body) in enumerate(
        chunk_text(document.text, max_chars=max_chars, overlap=overlap)
    ):
        chunks.append(
            Chunk(
                source_id=document.source_id,
                chunk_index=index,
                text=body,
                title=document.title,
                section=section,
                source_type=document.source_type,
                uri=document.uri,
                access_tags=list(document.access_tags),
            )
        )
    return chunks
