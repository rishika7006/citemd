"""PubMed ingestion via NCBI Entrez E-utilities.

Uses ``esearch`` to find PMIDs for a query and ``efetch`` to pull their titles and
abstracts. Only public abstract metadata is fetched (no full text, no PHI). The XML parser
is a pure function so it can be unit-tested against a fixture without network access.

NCBI asks callers to identify themselves and limits unauthenticated traffic to ~3 requests
per second. Set CITEMD via an API key in the environment if you need higher throughput.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx

from citemd.models import Document

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def parse_pubmed_xml(xml_text: str) -> list[Document]:
    """Parse an EFetch PubMed XML response into Documents (PMID, title, abstract)."""
    root = ET.fromstring(xml_text)
    docs: list[Document] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//MedlineCitation/PMID")
        pmid = pmid_el.text.strip() if pmid_el is not None and pmid_el.text else ""
        if not pmid:
            continue
        title_el = article.find(".//Article/ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Abstracts may have multiple labeled sections; join them with labels as headings.
        sections: list[str] = []
        for ab in article.findall(".//Article/Abstract/AbstractText"):
            label = ab.get("Label")
            body = "".join(ab.itertext()).strip()
            if not body:
                continue
            sections.append(f"# {label.title()}\n\n{body}" if label else body)
        abstract = "\n\n".join(sections)

        journal_el = article.find(".//Article/Journal/Title")
        journal = journal_el.text.strip() if journal_el is not None and journal_el.text else ""
        year_el = article.find(".//Article/Journal/JournalIssue/PubDate/Year")
        year = year_el.text.strip() if year_el is not None and year_el.text else ""

        text = f"{title}\n\n{abstract}".strip() if abstract else title
        docs.append(
            Document(
                source_id=f"pmid:{pmid}",
                title=title,
                text=text,
                source_type="pubmed",
                uri=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                metadata={"pmid": pmid, "journal": journal, "year": year},
            )
        )
    return docs


def search_pmids(
    query: str,
    *,
    max_results: int = 200,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> list[str]:
    """Return PMIDs matching a query via Entrez esearch."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": str(max_results),
        "retmode": "json",
        "tool": "citemd",
        "email": "noreply@example.com",
    }
    if api_key:
        params["api_key"] = api_key
    resp = httpx.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_documents(
    query: str,
    *,
    max_results: int = 200,
    api_key: str | None = None,
    timeout: float = 60.0,
) -> list[Document]:
    """Search PubMed and fetch abstracts as Documents."""
    pmids = search_pmids(query, max_results=max_results, api_key=api_key, timeout=timeout)
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": "citemd",
        "email": "noreply@example.com",
    }
    if api_key:
        params["api_key"] = api_key
    resp = httpx.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=timeout)
    resp.raise_for_status()
    return parse_pubmed_xml(resp.text)
