"""Document parsers for PDF, HTML, and Markdown sources.

Each parser returns a :class:`Document` with text and best-effort metadata. PDF parsing
preserves a page marker (a Markdown heading per page) so the chunker can attach page
numbers later. Parsers for binary/markup formats use the ``ingest`` extra and import lazily.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from citemd.models import Document

_PAGE_MARKER_RE = re.compile(r"^# page (\d+)$", re.MULTILINE)


def _source_id_for_path(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"file:{digest}"


def parse_markdown(path: str | Path) -> Document:
    """Parse a Markdown or plain-text file (no extra dependencies)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    title = p.stem
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    if first.startswith("#"):
        title = first.lstrip("#").strip() or title
    return Document(
        source_id=_source_id_for_path(p),
        title=title,
        text=text,
        source_type="markdown",
        uri=str(p),
    )


def parse_html(path: str | Path) -> Document:
    """Parse an HTML file into clean text using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "HTML parsing needs the 'ingest' extra. Install with: pip install 'citemd[ingest]'"
        ) from exc
    p = Path(path)
    soup = BeautifulSoup(p.read_text(encoding="utf-8", errors="replace"), "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else p.stem)
    text = "\n\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    return Document(
        source_id=_source_id_for_path(p),
        title=title,
        text=text,
        source_type="html",
        uri=str(p),
    )


def parse_pdf(path: str | Path) -> Document:
    """Parse a PDF, inserting a ``# page N`` heading before each page's text.

    The page headings let the section-aware chunker tag chunks with their page, which is
    the basis for precise multimodal citations in a later milestone.
    """
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "PDF parsing needs the 'ingest' extra. Install with: pip install 'citemd[ingest]'"
        ) from exc
    p = Path(path)
    reader = PdfReader(str(p))
    parts: list[str] = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            parts.append(f"# page {i}\n\n{page_text}")
    meta_title = ""
    if reader.metadata and reader.metadata.title:
        meta_title = str(reader.metadata.title).strip()
    return Document(
        source_id=_source_id_for_path(p),
        title=meta_title or p.stem,
        text="\n\n".join(parts),
        source_type="pdf",
        uri=str(p),
    )


def page_for_offset(text: str, offset: int) -> int | None:
    """Given the page-marked text and a character offset, return the page number.

    Helper for attaching page numbers to PDF-derived chunks.
    """
    page: int | None = None
    for m in _PAGE_MARKER_RE.finditer(text):
        if m.start() <= offset:
            page = int(m.group(1))
        else:
            break
    return page


def parse_file(path: str | Path) -> Document:
    """Dispatch to the right parser based on file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix in {".html", ".htm"}:
        return parse_html(path)
    if suffix in {".md", ".markdown", ".txt"}:
        return parse_markdown(path)
    raise ValueError(f"Unsupported file type: {suffix}")
