import pytest

from citemd.ingest.parse import page_for_offset, parse_file, parse_markdown


def test_parse_markdown_uses_first_heading_as_title(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("# Clinical Note\n\nSome body text.", encoding="utf-8")
    doc = parse_markdown(p)
    assert doc.title == "Clinical Note"
    assert doc.source_type == "markdown"
    assert "body text" in doc.text


def test_parse_file_dispatch_unknown_extension(tmp_path):
    p = tmp_path / "data.xyz"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_file(p)


def test_page_for_offset():
    text = "# page 1\n\nalpha content\n\n# page 2\n\nbeta content"
    beta_offset = text.index("beta")
    alpha_offset = text.index("alpha")
    assert page_for_offset(text, alpha_offset) == 1
    assert page_for_offset(text, beta_offset) == 2
    assert page_for_offset("no markers here", 3) is None
