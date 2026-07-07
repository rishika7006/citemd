"""Build a broad PubMed corpus for evaluation.

Fetches abstracts across many clinical topics and loads each topic into the hybrid index
immediately, so progress persists to the database as it goes and the job is resumable: a
completed-topics marker file lets a re-run skip topics already loaded. One embedding-model
load is shared across all topics. Public abstract metadata only; no PHI.

    python -u scripts/build_corpus.py [per_topic]

Re-running is safe and resumes: documents upsert by source_id and finished topics are skipped.
"""

import sys
import time
from pathlib import Path

from citemd.config import get_settings
from citemd.ingest.embed import Embedder
from citemd.ingest.loader import load_documents
from citemd.ingest.pubmed import fetch_documents

# Broad coverage across major specialties so retrieval has something to find for the
# literature-grounded MIRAGE datasets (PubMedQA, BioASQ) and general medical questions.
TOPICS = [
    "type 2 diabetes mellitus treatment", "hypertension management", "heart failure therapy",
    "myocardial infarction", "atrial fibrillation anticoagulation", "ischemic stroke",
    "chronic obstructive pulmonary disease", "asthma management", "community acquired pneumonia",
    "sepsis management", "chronic kidney disease", "acute kidney injury", "viral hepatitis",
    "liver cirrhosis", "inflammatory bowel disease", "peptic ulcer disease", "anemia diagnosis",
    "acute leukemia", "lymphoma treatment", "breast cancer therapy", "non small cell lung cancer",
    "colorectal cancer screening", "prostate cancer management", "major depressive disorder",
    "schizophrenia treatment", "Alzheimer disease", "Parkinson disease", "epilepsy treatment",
    "multiple sclerosis", "rheumatoid arthritis", "systemic lupus erythematosus", "osteoporosis",
    "hypothyroidism", "hyperthyroidism", "tuberculosis treatment", "HIV antiretroviral therapy",
    "COVID-19 clinical management", "antibiotic resistance", "vaccine efficacy",
    "pulmonary embolism", "venous thromboembolism", "gestational diabetes", "preeclampsia",
    "pediatric asthma", "sepsis biomarkers", "diabetic nephropathy", "coronary artery disease",
    "obesity management", "thyroid nodule", "migraine prophylaxis",
]


def main() -> int:
    per_topic = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    marker = Path(get_settings().artifacts_dir) / "corpus_topics_done.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    done = set(marker.read_text().split("\n")) if marker.exists() else set()
    done.discard("")

    remaining = [t for t in TOPICS if t not in done]
    print(
        f"{len(done)} topics already loaded; {len(remaining)} remaining. "
        f"Fetching up to {per_topic} abstracts each and loading per topic...",
        flush=True,
    )

    embedder = Embedder()  # one model load, shared across topics
    total_docs = 0
    total_chunks = 0
    t0 = time.time()
    for i, topic in enumerate(remaining, 1):
        try:
            docs = fetch_documents(topic, max_results=per_topic)
            counts = load_documents(docs, embedder=embedder)
        except Exception as exc:  # a network/DB hiccup on one topic should not kill the run
            print(f"  [{i}/{len(remaining)}] {topic!r}: ERROR {exc}", flush=True)
            continue
        total_docs += counts["documents"]
        total_chunks += counts["chunks"]
        with marker.open("a", encoding="utf-8") as fh:
            fh.write(topic + "\n")
        print(
            f"  [{i}/{len(remaining)}] {topic!r}: +{counts['documents']} docs, "
            f"+{counts['chunks']} chunks (session totals {total_docs} docs, {total_chunks} "
            f"chunks, {time.time() - t0:.0f}s)",
            flush=True,
        )

    print(f"\nDONE this session: {total_docs} documents, {total_chunks} chunks.", flush=True)
    print("CORPUS_BUILD_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
