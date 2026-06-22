from citemd.ingest.pubmed import parse_pubmed_xml

SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <Journal>
          <Title>Journal of Test Medicine</Title>
          <JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue>
        </Journal>
        <ArticleTitle>Effect of metformin on outcomes</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Metformin is widely used.</AbstractText>
          <AbstractText Label="RESULTS">It reduced events.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_pubmed_basic():
    docs = parse_pubmed_xml(SAMPLE_XML)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.source_id == "pmid:12345"
    assert doc.title == "Effect of metformin on outcomes"
    assert doc.source_type == "pubmed"
    assert doc.uri == "https://pubmed.ncbi.nlm.nih.gov/12345/"
    assert doc.metadata["journal"] == "Journal of Test Medicine"
    assert doc.metadata["year"] == "2021"
    # Labeled abstract sections become Markdown headings the chunker understands.
    assert "# Background" in doc.text
    assert "# Results" in doc.text


def test_parse_pubmed_skips_entries_without_pmid():
    xml = "<PubmedArticleSet><PubmedArticle><MedlineCitation>"
    xml += "<Article><ArticleTitle>No id</ArticleTitle></Article>"
    xml += "</MedlineCitation></PubmedArticle></PubmedArticleSet>"
    assert parse_pubmed_xml(xml) == []
