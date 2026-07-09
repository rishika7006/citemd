import { LINKS } from "@/lib/links";
import { GithubIcon, LinkedinIcon, MailIcon } from "@/components/icons";

const STAGES: { title: string; body: string }[] = [
  {
    title: "Ingest and index",
    body: "PubMed abstracts are parsed, chunked, and embedded on CPU, then stored in Postgres with pgvector for dense search and full-text BM25 for keywords.",
  },
  {
    title: "Retrieve and read",
    body: "A question hits both indexes, fused with Reciprocal Rank Fusion and reranked by a cross-encoder. A model writes the answer from those passages with a citation on every claim.",
  },
  {
    title: "Abstain",
    body: "A confidence gate declines the least-supported questions instead of guessing, so a wrong-but-confident answer is turned into a measured abstention.",
  },
  {
    title: "Serve and evaluate",
    body: "Generation runs through KVGate to a provider or to self-hosted vLLM with LMCache. MIRAGE questions score accuracy, calibration, faithfulness, and risk-coverage.",
  },
];

const STACK = [
  "Python",
  "Postgres",
  "pgvector",
  "BGE embeddings",
  "bge-reranker",
  "Claude Haiku 4.5",
  "KVGate",
  "vLLM",
  "LMCache",
  "Redis",
  "FastAPI",
  "Next.js",
  "TypeScript",
  "Docker",
  "GitHub Actions",
];

function ArchitectureAndStack() {
  return (
    <section className="mt-20 border-t border-line pt-12">
      <div className="text-xs uppercase tracking-wider text-accent mb-1">how it works</div>
      <h2 className="font-serif text-2xl mb-6">Architecture and stack</h2>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {STAGES.map((s, i) => (
          <div key={s.title} className="rounded-md border border-line bg-raised p-4">
            <div className="flex items-baseline gap-2 mb-2">
              <span className="font-mono text-xs text-muted">{i + 1}</span>
              <h3 className="font-medium">{s.title}</h3>
            </div>
            <p className="text-sm text-muted leading-relaxed">{s.body}</p>
          </div>
        ))}
      </div>

      <div className="mt-6 flex flex-wrap gap-2">
        {STACK.map((t) => (
          <span
            key={t}
            className="rounded-full border border-line bg-sunk px-3 py-1 font-mono text-xs text-muted"
          >
            {t}
          </span>
        ))}
      </div>
    </section>
  );
}

function Closing() {
  return (
    <section className="mt-16 rounded-md border border-line bg-raised px-6 py-10 text-center">
      <h2 className="font-serif text-2xl mb-3">Open source and reproducible</h2>
      <p className="mx-auto max-w-reading text-sm text-muted leading-relaxed">
        CiteMD is an open-source clinical-evidence RAG with a full evaluation harness. Every number
        on this site is measured and reproducible from the commands in the repository. Built by
        Rishika Vaish.
      </p>
      <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
        <a
          href={LINKS.repo}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-md border border-accent bg-accent-soft/40 px-4 py-2 text-sm text-ink hover:bg-accent-soft transition-colors"
        >
          <GithubIcon className="w-4 h-4" />
          View on GitHub
        </a>
        <a
          href={LINKS.linkedin}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-2 rounded-md border border-line px-4 py-2 text-sm text-muted hover:text-ink transition-colors"
        >
          <LinkedinIcon className="w-4 h-4" />
          LinkedIn
        </a>
        <a
          href={`mailto:${LINKS.email}`}
          className="inline-flex items-center gap-2 rounded-md border border-line px-4 py-2 text-sm text-muted hover:text-ink transition-colors"
        >
          <MailIcon className="w-4 h-4" />
          {LINKS.email}
        </a>
      </div>
    </section>
  );
}

export function AboutSections() {
  return (
    <>
      <ArchitectureAndStack />
      <Closing />
    </>
  );
}
