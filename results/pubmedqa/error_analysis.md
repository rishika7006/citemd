# PubMedQA error analysis (n = 300, hybrid arm)

A short look at the 75 wrong answers (25.0% of 300), to say what the system got wrong and why.

## Where the errors come from

| Category | Count |
|----------|-------|
| Evidence not retrieved (retrieval miss) | 0 |
| Evidence retrieved but wrong option (reader error) | 75 |
| Of the wrong answers, low-confidence (< 0.50) | 10 |

Two things stand out.

1. **Retrieval is not the bottleneck.** Every wrong answer had the gold source abstract in the
   retrieved context (hit-rate is about 99% overall). None of the errors are retrieval misses.
   The system is failing at reading and reasoning over evidence it already has, not at finding
   evidence. This is consistent with the ablation, where the retrieval variants do not separate.

2. **Most errors are confident errors.** Only 10 of 75 wrong answers were below 0.50 confidence,
   so abstention on the least-confident questions removes some but not most of them. This is why
   abstaining on the riskiest 20% reduces error by about 28% relative rather than eliminating it,
   and it matches the reliability diagram, which shows the model is overconfident in the middle
   of the confidence range.

## What the confident errors look like

A recurring pattern: the gold answer is "maybe" (option C), the abstract hedges, and the model
commits to a definite "yes" or "no" at high confidence. Examples (gold, predicted, confidence):

- gold C, predicted A, conf 0.95: antral follicle assessment as a predictor of IVF outcome.
- gold C, predicted B, conf 0.95: specialised phonological-awareness training in preschoolers.
- gold C, predicted A, conf 0.95: sternal fracture in growing children.
- gold B, predicted A, conf 0.95: renal warm ischemia over 30 minutes during partial nephrectomy.

## What the system learned, and what it did not

- It learned to retrieve the right source and to cite it. Retrieval and grounding are solved for
  this dataset.
- It did not learn calibrated uncertainty on genuinely equivocal questions. The largest error
  mode is confident commitment to yes or no where the evidence supports "maybe." Better
  handling of that ambiguity, not better retrieval, is where accuracy would come from next.
