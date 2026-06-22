# Examples

## Labeled retrieval set (`labels.example.jsonl`)

A labeled retrieval set scores the retriever with `citemd eval retrieval`. Each line is a
JSON object with a `query` and the `relevant_ids` that should be retrieved for it:

```json
{"query": "Does metformin reduce cardiovascular mortality in type 2 diabetes?", "relevant_ids": ["pmid:34567890#0"]}
```

`relevant_ids` are chunk ids (`<source_id>#<chunk_index>`) by default, or `source_id`
values when you pass `--by-source`. Replace the placeholder ids with real ones from your
ingested corpus, then run:

```bash
citemd eval retrieval --labels examples/labels.example.jsonl --k 10 --retriever hybrid
citemd eval retrieval --labels examples/labels.example.jsonl --k 10 --retriever vector
citemd eval retrieval --labels examples/labels.example.jsonl --k 10 --retriever bm25
```

Comparing the three retrievers on the same labels is the seed of the milestone-1 ablation
(vector-only versus hybrid). End-to-end QA evaluation on MIRAGE is milestone 2.
