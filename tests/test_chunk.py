from citemd.ingest.chunk import chunk_document, chunk_text
from citemd.models import Document


def test_chunk_respects_max_chars():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    chunks = chunk_text(text, max_chars=300, overlap=20)
    assert chunks, "expected at least one chunk"
    for _section, body in chunks:
        assert len(body) <= 300 + 20  # overlap may push slightly past the target


def test_chunk_is_section_aware():
    text = (
        "# Background\n\nThis is the background section with some content here.\n\n"
        "# Methods\n\nWe did the methods this way, carefully and thoroughly.\n\n"
        "# Results\n\nThe results were measured and reported honestly."
    )
    chunks = chunk_text(text, max_chars=1000, overlap=0)
    sections = {section for section, _ in chunks}
    assert sections == {"Background", "Methods", "Results"}


def test_hard_split_of_long_paragraph():
    text = "word " * 500  # one giant paragraph, no blank lines
    chunks = chunk_text(text, max_chars=200, overlap=0)
    assert len(chunks) > 1
    for _section, body in chunks:
        assert len(body) <= 200


def test_chunk_document_carries_metadata():
    doc = Document(
        source_id="pmid:1",
        title="A Study",
        text="# Intro\n\nSome introductory text about a clinical topic.",
        source_type="pubmed",
        uri="https://pubmed.ncbi.nlm.nih.gov/1/",
        access_tags=["public"],
    )
    chunks = chunk_document(doc, max_chars=500, overlap=0)
    assert chunks
    first = chunks[0]
    assert first.source_id == "pmid:1"
    assert first.title == "A Study"
    assert first.section == "Intro"
    assert first.chunk_id == "pmid:1#0"
    assert first.access_tags == ["public"]


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n\n  ") == []
