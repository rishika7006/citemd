import { LINKS } from "@/lib/links";

type Item = { id: string; label: string };
type Group = { group: string; items: Item[] };

const NAV: Group[] = [
  {
    group: "Getting started",
    items: [
      { id: "introduction", label: "Introduction" },
      { id: "quickstart", label: "Quickstart" },
      { id: "configuration", label: "Configuration" },
    ],
  },
  {
    group: "Pipeline",
    items: [
      { id: "ingestion", label: "Ingestion and index" },
      { id: "retrieval", label: "Hybrid retrieval" },
      { id: "reading", label: "Reading and abstention" },
    ],
  },
  {
    group: "Evaluation",
    items: [
      { id: "datasets", label: "Datasets" },
      { id: "metrics", label: "Metrics" },
      { id: "reproduce", label: "Reproduce results" },
    ],
  },
  {
    group: "Serving",
    items: [
      { id: "kvgate", label: "KVGate integration" },
      { id: "lmcache", label: "vLLM and LMCache" },
    ],
  },
  {
    group: "Reference",
    items: [
      { id: "cli", label: "CLI commands" },
      { id: "safety", label: "Data and safety" },
      { id: "limitations", label: "Limitations" },
    ],
  },
];

function Code({ children }: { children: string }) {
  return (
    <pre className="mt-3 rounded-md border border-line bg-sunk px-4 py-3 font-mono text-xs leading-relaxed overflow-x-auto text-ink">
      {children}
    </pre>
  );
}

function H({ id, kicker, children }: { id: string; kicker: string; children: React.ReactNode }) {
  return (
    <div className="mb-3" style={{ scrollMarginTop: "1.5rem" }} id={id}>
      <div className="text-xs uppercase tracking-wider text-accent mb-1">{kicker}</div>
      <h2 className="font-serif text-2xl">{children}</h2>
    </div>
  );
}

function P({ children }: { children: React.ReactNode }) {
  return <p className="text-[15px] leading-relaxed text-muted">{children}</p>;
}

function Mono({ children }: { children: React.ReactNode }) {
  return <code className="font-mono text-[13px] text-ink">{children}</code>;
}

function Sidebar() {
  return (
    <nav className="hidden lg:block sticky top-6 self-start text-sm">
      {NAV.map((g) => (
        <div key={g.group} className="mb-5">
          <div className="text-xs uppercase tracking-wider text-muted mb-2">{g.group}</div>
          <ul className="space-y-1.5">
            {g.items.map((it) => (
              <li key={it.id}>
                <a href={`#${it.id}`} className="text-muted hover:text-ink transition-colors">
                  {it.label}
                </a>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  );
}

export function DocsView() {
  return (
    <div className="grid lg:grid-cols-[190px_minmax(0,1fr)] gap-12 items-start">
      <Sidebar />
      <div className="max-w-reading space-y-14">
        <section>
          <H id="introduction" kicker="getting started">Introduction</H>
          <div className="space-y-3">
            <P>
              CiteMD is an open-source clinical-evidence RAG system. It answers medical research
              questions from published literature with a citation on every claim, and it abstains
              when the retrieved evidence is too weak to answer safely. The emphasis is measured
              trustworthiness: abstention, calibration, and citation faithfulness, all reported with
              confidence intervals and honest negative results.
            </P>
            <P>
              This is not a novel algorithm. It is a rigorously evaluated, reproducible artifact
              that evaluates on the MIRAGE benchmark, the recognized standard for medical RAG.
            </P>
            <div className="rounded-md border border-warm/40 bg-warm/5 px-4 py-3 text-sm text-muted">
              Research and evaluation only. CiteMD is not a clinical tool and must not be used for
              diagnosis, treatment, or any patient-care decision. Public, non-PHI data only.
            </div>
          </div>
        </section>

        <section>
          <H id="quickstart" kicker="getting started">Quickstart</H>
          <div className="space-y-3">
            <P>
              The web app runs in demo mode with precomputed answers, no API key and no model call:
            </P>
            <Code>{`git clone ${LINKS.repo}.git
cd citemd/web
npm install
npm run dev            # http://localhost:3000`}</Code>
            <P>
              The Python package uses a light core so the pure-logic parts install and test with no
              GPU or database. Heavier capabilities are optional extras:
            </P>
            <Code>{`pip install -e ".[dev]"    # test and lint tooling (light core)
pip install -e ".[all]"    # Postgres, embeddings, parsers, LLM client, charts`}</Code>
          </div>
        </section>

        <section>
          <H id="configuration" kicker="getting started">Configuration</H>
          <div className="space-y-3">
            <P>
              Generation needs an OpenAI-compatible backend. Copy <Mono>.env.example</Mono> to{" "}
              <Mono>.env</Mono> and set three variables. Point them at a provider directly, or at
              KVGate, with no application code change:
            </P>
            <Code>{`CITEMD_LLM_BASE_URL=http://localhost:8080/v1   # KVGate, or a provider URL
CITEMD_LLM_API_KEY=...                          # or "not-needed" for a local gateway
CITEMD_LLM_MODEL=claude-haiku-4-5               # a logical model name`}</Code>
            <P>
              The optional extras are <Mono>db</Mono> (Postgres, pgvector), <Mono>ml</Mono>{" "}
              (embeddings and reranker), <Mono>ingest</Mono> (PDF, HTML, Markdown parsers),{" "}
              <Mono>llm</Mono> (generation client), and <Mono>viz</Mono> (charts).
            </P>
          </div>
        </section>

        <section>
          <H id="ingestion" kicker="pipeline">Ingestion and index</H>
          <div className="space-y-3">
            <P>
              Documents are parsed and chunked with source, page, and section metadata, embedded on
              CPU with a BGE model, and stored in a hybrid index. Dense vectors live in Postgres via
              pgvector; the same text is indexed for Postgres full-text BM25. One store backs both
              retrieval modes.
            </P>
            <Code>{`docker compose up -d && citemd db init
citemd ingest pubmed --query "type 2 diabetes management" --max 300`}</Code>
          </div>
        </section>

        <section>
          <H id="retrieval" kicker="pipeline">Hybrid retrieval</H>
          <div className="space-y-3">
            <P>
              A question is retrieved against the dense and BM25 indexes in parallel. The two ranked
              lists are combined with Reciprocal Rank Fusion, then a cross-encoder
              (bge-reranker-base) reranks the fused candidates so the most relevant passages surface
              in the top k.
            </P>
            <Code>{`citemd query "What is first-line treatment for type 2 diabetes?" --k 5`}</Code>
          </div>
        </section>

        <section>
          <H id="reading" kicker="pipeline">Reading and abstention</H>
          <div className="space-y-3">
            <P>
              The reader is prompted to answer using only the retrieved passages and to attach an
              inline citation to every claim. The answer carries a self-reported confidence, which a
              gate compares against a threshold. Below it, the system abstains rather than return a
              confident wrong answer. Abstention is a first-class outcome, measured, not hidden.
            </P>
            <Code>{`citemd ask "Does metformin reduce cardiovascular mortality in type 2 diabetes?"`}</Code>
          </div>
        </section>

        <section>
          <H id="datasets" kicker="evaluation">Datasets</H>
          <div className="space-y-3">
            <P>
              Evaluation uses MIRAGE (MedQA, PubMedQA, BioASQ, MMLU-Med, MedMCQA). The headline
              results run on a 300-question seeded PubMedQA sample. The corpus is about 13,700 PubMed
              abstracts: roughly 13.2k topic abstracts as distractors plus the PubMedQA source
              abstracts, fetched by PMID, so retrieval has the gold evidence to find.
            </P>
            <P>
              The MIRAGE files are class-ordered, so evaluation uses <Mono>--sample</Mono> (a seeded
              random subset), not <Mono>--limit</Mono> (the first N in file order), for a
              representative measurement.
            </P>
          </div>
        </section>

        <section>
          <H id="metrics" kicker="evaluation">Metrics</H>
          <div className="space-y-3">
            <P>Each run reports:</P>
            <ul className="space-y-2 text-[15px] text-muted leading-relaxed list-disc pl-5">
              <li>Accuracy with 95% Wilson confidence intervals, per retrieval arm.</li>
              <li>Retrieval hit-rate: whether the gold source reached the top k.</li>
              <li>Calibration (expected calibration error) and a reliability diagram.</li>
              <li>
                Risk-coverage: error among answered questions as the abstention threshold moves.
              </li>
              <li>
                Citation faithfulness: an LLM judge checks whether the cited passages actually
                support the answer, not just that a citation is present.
              </li>
            </ul>
            <P>
              On the PubMedQA sample, abstaining on the riskiest 20% cut error from 25.0% to 17.9%,
              about a 28% relative reduction, while still answering 80% of questions, with expected
              calibration error 0.067.
            </P>
          </div>
        </section>

        <section>
          <H id="reproduce" kicker="evaluation">Reproduce results</H>
          <div className="space-y-3">
            <P>Runs cache per question and resume, so a long evaluation can be interrupted safely.</P>
            <Code>{`# Ingest the PubMedQA source abstracts among the distractors
citemd ingest mirage-sources --datasets pubmedqa

# Ablation with confidence intervals, hit-rate, calibration, and charts
citemd eval ablation --dataset pubmedqa --sample 300 --seed 0 --charts

# Citation faithfulness (LLM-judge support scoring)
citemd eval faithfulness --dataset pubmedqa --sample 60 --seed 0`}</Code>
          </div>
        </section>

        <section>
          <H id="kvgate" kicker="serving">KVGate integration</H>
          <div className="space-y-3">
            <P>
              Because CiteMD's LLM client speaks the OpenAI protocol, it points at{" "}
              <a href={LINKS.kvgate} target="_blank" rel="noreferrer" className="text-accent hover:underline">
                KVGate
              </a>{" "}
              by changing the three environment variables above. Through KVGate, generation gains
              exact and semantic response caching, token-bucket rate limiting, per-tenant budgets,
              provider failover, one endpoint, and metrics, with no application code change.
            </P>
            <P>
              On the hosted path, an identical repeat request is served from the response cache with
              no model call, turning a real call into a 2 ms cache hit. The Workstation footer strip
              shows the live cache status, latency, and cost per answer.
            </P>
          </div>
        </section>

        <section>
          <H id="lmcache" kicker="serving">vLLM and LMCache</H>
          <div className="space-y-3">
            <P>
              For self-hosted serving, KVGate sits in front of vLLM replicas with LMCache in MP
              mode (CPU L1 plus Redis L2). CiteMD was benchmarked against a real multimodal clinical
              workload (48 open-access radiology figures) on 2x A40:
            </P>
            <ul className="space-y-2 text-[15px] text-muted leading-relaxed list-disc pl-5">
              <li>
                Prefix-aware routing cut TTFT p50 by 31% and tail latency p95 by 47%, and raised
                throughput by 54%, at 99.4% routing affinity.
              </li>
              <li>
                Under GPU memory pressure, CPU L1 cut TTFT p50 by 33%; adding Redis L2 cut tail
                latency p95 by 65% and raised throughput by 31%.
              </li>
              <li>
                When the KV fits GPU memory, LMCache is net overhead. It is worth enabling by the
                working-set-to-GPU ratio, not by default. The Serving Path tab shows this crossover.
              </li>
            </ul>
          </div>
        </section>

        <section>
          <H id="cli" kicker="reference">CLI commands</H>
          <div className="space-y-3">
            <Code>{`citemd db init                                   # schema on Postgres + pgvector
citemd ingest pubmed --query "..." --max 300     # ingest abstracts
citemd ingest mirage-sources --datasets pubmedqa # gold sources for eval
citemd query "..." --k 5                          # retrieve only
citemd ask "..."                                  # retrieve, read, abstain
citemd eval fetch                                 # download MIRAGE datasets
citemd eval ablation --dataset pubmedqa --sample 300 --seed 0 --charts
citemd eval faithfulness --dataset pubmedqa --sample 60 --seed 0`}</Code>
          </div>
        </section>

        <section>
          <H id="safety" kicker="reference">Data and safety</H>
          <div className="space-y-3">
            <ul className="space-y-2 text-[15px] text-muted leading-relaxed list-disc pl-5">
              <li>Public, non-PHI data only. No real patient records, no credentialed datasets.</li>
              <li>MIRAGE for evaluation; PubMed abstracts for the corpus.</li>
              <li>Research and evaluation only. Not for clinical use.</li>
            </ul>
          </div>
        </section>

        <section>
          <H id="limitations" kicker="reference">Limitations</H>
          <div className="space-y-3">
            <ul className="space-y-2 text-[15px] text-muted leading-relaxed list-disc pl-5">
              <li>
                Results are on one dataset (PubMedQA), n=300, one generation model. Absolute
                accuracy will shift with a different model.
              </li>
              <li>
                Because the gold abstract is in the corpus, the eval measures reading and abstention
                given retrievable evidence, not open-web retrieval difficulty.
              </li>
              <li>
                Confidence is the model's self-reported value; reasonably calibrated here but not a
                guaranteed probability.
              </li>
              <li>
                Citation faithfulness uses an LLM judge; using the same model to judge its own
                citations is a known bias.
              </li>
              <li>
                A stronger reader (Claude Sonnet 4.6) was tested and did not help: within noise on
                accuracy and worse calibrated, so CiteMD uses Haiku 4.5.
              </li>
              <li>
                Figure-grounded answering was tested and not shipped: a general vision model leaned
                on the figure caption rather than reading the image, so shipping it would overclaim.
              </li>
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
}
