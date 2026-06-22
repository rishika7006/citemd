"""Manual smoke check for DB-backed retrieval. Not part of the test suite."""

from citemd.ingest.embed import Embedder
from citemd.retrieve.bm25 import bm25_search
from citemd.retrieve.hybrid import hybrid_search
from citemd.retrieve.vector import vector_search

emb = Embedder()
emb.embed_query("warmup")
q = "Does metformin reduce cardiovascular mortality in type 2 diabetes?"
for name, fn in [
    ("vector", lambda: vector_search(q, k=3, embedder=emb)),
    ("bm25", lambda: bm25_search(q, k=3)),
    ("hybrid", lambda: hybrid_search(q, k=3, embedder=emb)),
]:
    res = fn()
    print(f"\n== {name} ({len(res)} hits) ==", flush=True)
    for i, c in enumerate(res, 1):
        section = (c.section or "")[:18]
        snippet = c.text[:80].strip()
        print(f"[{i}] {c.score:.4f} {c.source_id} | {section:18} | {snippet}", flush=True)
print("\nSMOKE_OK", flush=True)
