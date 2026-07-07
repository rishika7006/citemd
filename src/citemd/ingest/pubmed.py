"""PubMed ingestion via NCBI Entrez E-utilities.

Uses ``esearch`` to find PMIDs for a query and ``efetch`` to pull their titles and
abstracts. Only public abstract metadata is fetched (no full text, no PHI). The XML parser
is a pure function so it can be unit-tested against a fixture without network access.

NCBI asks callers to identify themselves and limits unauthenticated traffic to ~3 requests
per second. Set CITEMD via an API key in the environment if you need higher throughput.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET

import httpx

from citemd.models import Document

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# NCBI allows ~3 requests/sec unauthenticated, ~10/sec with an API key. efetch URLs also
# have a practical length limit, so PMIDs are fetched in batches rather than one giant call.
_EFETCH_BATCH = 200


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


def _efetch(pmids: list[str], *, api_key: str | None, timeout: float) -> list[Document]:
    """Fetch one batch of PMIDs via efetch and parse them into Documents."""
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


def fetch_by_pmids(
    pmids: list[str],
    *,
    api_key: str | None = None,
    timeout: float = 60.0,
    delay: float = 0.34,
) -> list[Document]:
    """Fetch abstracts for an explicit list of PMIDs, batching efetch politely.

    Used to ingest the source literature behind PubMedQA/BioASQ questions so retrieval has
    the gold evidence to find (mirroring MedRAG-style retrieval over a full PubMed snapshot).
    """
    unique = list(dict.fromkeys(str(p) for p in pmids if str(p)))
    docs: list[Document] = []
    for start in range(0, len(unique), _EFETCH_BATCH):
        batch = unique[start : start + _EFETCH_BATCH]
        docs.extend(_efetch(batch, api_key=api_key, timeout=timeout))
        if start + _EFETCH_BATCH < len(unique):
            time.sleep(delay)
    return docs


def fetch_documents(
    query: str,
    *,
    max_results: int = 200,
    api_key: str | None = None,
    timeout: float = 60.0,
    delay: float = 0.34,
) -> list[Document]:
    """Search PubMed and fetch abstracts as Documents, batching efetch for large queries.

    ``delay`` is the courtesy pause between requests to stay under NCBI's rate limit; with an
    API key it can safely be lowered.
    """
    pmids = search_pmids(query, max_results=max_results, api_key=api_key, timeout=timeout)
    if not pmids:
        return []
    docs: list[Document] = []
    for start in range(0, len(pmids), _EFETCH_BATCH):
        batch = pmids[start : start + _EFETCH_BATCH]
        docs.extend(_efetch(batch, api_key=api_key, timeout=timeout))
        if start + _EFETCH_BATCH < len(pmids):
            time.sleep(delay)
    return docs
