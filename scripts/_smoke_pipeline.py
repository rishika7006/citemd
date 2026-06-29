"""Free end-to-end pipeline smoke: real hybrid retrieval + real cross-encoder rerank +
a fake LLM (no API spend). Proves the reranker runs on real data and the pipeline wires
contexts/citations correctly. Not part of the test suite."""

from citemd.generate.client import FakeLLMClient
from citemd.generate.prompt import format_contexts
from citemd.pipeline import PipelineConfig, answer


def fake_responder(messages):
    # Echo which passages exist, then "cite" the first two as a confident answer.
    return '{"answer": "B", "confidence": 0.83, "citations": [1, 2], "rationale": "smoke"}'


q = "Does metformin reduce cardiovascular mortality in type 2 diabetes?"
cfg = PipelineConfig(retriever="hybrid", use_rerank=True, candidate_k=20, top_k=4)
llm = FakeLLMClient(fake_responder, model="fake")
res = answer(q, llm, options={"A": "no", "B": "yes"}, config=cfg)

print("RERANKED CONTEXTS (top score first):", flush=True)
for c in res.contexts:
    print(f"  {c.score:+.3f} {c.source_id} | {c.text[:70].strip()}", flush=True)
print(f"\nanswer option = {res.option}", flush=True)
print(f"abstained     = {res.abstained}", flush=True)
print(f"confidence    = {res.confidence}", flush=True)
print("citations     =", [(c.marker, c.source_id) for c in res.citations], flush=True)
print("\nPROMPT PREVIEW (first 300 chars of numbered passages):", flush=True)
print(format_contexts(res.contexts)[:300], flush=True)
print("\nPIPELINE_SMOKE_OK", flush=True)
