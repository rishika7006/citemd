"use client";

import { useMemo, useState } from "react";
import type { AnswerRecord, Context } from "@/lib/types";

function pct(x: number) {
  return `${Math.round(x * 100)}%`;
}

function ConfidenceMeter({ value, abstained }: { value: number; abstained: boolean }) {
  return (
    <div className="w-full">
      <div className="flex items-baseline gap-2 mb-1.5">
        <span className="text-xs uppercase tracking-wider text-muted">conf</span>
        <span className="font-mono text-sm tnum">{value.toFixed(2)}</span>
      </div>
      <div className="relative h-1.5 bg-sunk rounded-full overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{
            width: pct(value),
            background: abstained ? "var(--warm)" : "var(--accent)",
          }}
        />
      </div>
    </div>
  );
}

function DecisionBar({ a }: { a: AnswerRecord }) {
  const answerLabel = a.predicted ? a.options[a.predicted] : null;
  return (
    <div className="border border-line rounded-md bg-raised">
      <div className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-line">
        <div className="p-4">
          <div className="text-xs uppercase tracking-wider text-muted mb-1">decision</div>
          {a.abstained ? (
            <div className="font-serif text-lg" style={{ color: "var(--warm)" }}>
              Abstained
            </div>
          ) : (
            <div className="font-serif text-lg">
              {a.predicted}
              {answerLabel ? <span className="text-muted text-base"> · {answerLabel}</span> : null}
            </div>
          )}
        </div>
        <div className="p-4">
          <div className="text-xs uppercase tracking-wider text-muted mb-1">retrieval</div>
          <div className="font-mono text-sm">{a.retrieval_mode}</div>
        </div>
        <div className="p-4">
          <div className="text-xs uppercase tracking-wider text-muted mb-1">gold source</div>
          <div className="text-sm">
            {a.gold_retrieved ? (
              <span>
                retrieved <span className="text-muted">· rank {a.gold_rank}</span>
              </span>
            ) : (
              <span style={{ color: "var(--warm)" }}>not retrieved</span>
            )}
          </div>
        </div>
        <div className="p-4 flex items-center">
          <ConfidenceMeter value={a.confidence} abstained={a.abstained} />
        </div>
      </div>
    </div>
  );
}

function KvgateStrip({ a }: { a: AnswerRecord }) {
  if (!a.kvgate.enabled) return null;
  const cells: [string, string][] = [
    ["gateway", "KVGate"],
    ["provider", a.kvgate.provider],
    ["model", a.kvgate.model],
    ["cache", a.kvgate.cache],
    ["latency", `${a.kvgate.latency_ms} ms`],
    ["cost", a.kvgate.cost_usd === 0 ? "0 (cached)" : `$${a.kvgate.cost_usd.toFixed(4)}`],
  ];
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-1 border border-line rounded-md bg-sunk px-4 py-2 font-mono text-xs">
      {cells.map(([k, v]) => (
        <span key={k} className="text-muted">
          {k} <span className="text-ink">{v}</span>
        </span>
      ))}
    </div>
  );
}

function AnswerProse({
  text,
  onCite,
  active,
}: {
  text: string;
  onCite: (n: number) => void;
  active: number | null;
}) {
  const parts = useMemo(() => text.split(/(\[\d+\])/g), [text]);
  return (
    <p className="font-serif text-[1.28rem] leading-relaxed max-w-reading">
      {parts.map((p, i) => {
        const m = p.match(/^\[(\d+)\]$/);
        if (m) {
          const n = Number(m[1]);
          return (
            <sup
              key={i}
              className="cite"
              onClick={() => onCite(n)}
              style={active === n ? { textDecoration: "underline" } : undefined}
            >
              [{n}]
            </sup>
          );
        }
        return <span key={i}>{p}</span>;
      })}
    </p>
  );
}

function Passage({ c, active }: { c: Context; active: boolean }) {
  const body = useMemo(() => {
    if (!c.support || !c.text.includes(c.support)) return [c.text];
    const idx = c.text.indexOf(c.support);
    return [c.text.slice(0, idx), c.support, c.text.slice(idx + c.support.length)];
  }, [c]);
  return (
    <div
      id={`passage-${c.marker}`}
      className={
        "rounded-md border p-4 transition-colors " +
        (active
          ? "border-accent bg-accent-soft/40"
          : c.cited
            ? "border-line bg-raised"
            : "border-line/60 bg-transparent")
      }
    >
      <div className="flex items-baseline justify-between gap-3 mb-1.5">
        <div className="flex items-baseline gap-2 min-w-0">
          <span className="font-mono text-xs text-accent">[{c.marker}]</span>
          <span className="text-sm font-medium truncate">{c.title}</span>
        </div>
        <span className="font-mono text-[11px] text-muted tnum shrink-0">
          {c.score.toFixed(2)}
        </span>
      </div>
      <div className="flex items-center gap-2 mb-2 font-mono text-[11px] text-muted">
        <span>{c.source_id}</span>
        {c.section ? <span>· {c.section}</span> : null}
        {c.cited ? <span className="text-accent">· cited</span> : null}
      </div>
      <p className="text-sm leading-relaxed text-muted">
        {body.length === 3 ? (
          <>
            {body[0]}
            <mark className="support">{body[1]}</mark>
            {body[2]}
          </>
        ) : (
          body[0]
        )}
      </p>
    </div>
  );
}

export function Workstation({ answers }: { answers: AnswerRecord[] }) {
  const [idx, setIdx] = useState(0);
  const [active, setActive] = useState<number | null>(null);
  const a = answers[idx];

  const onCite = (n: number) => {
    setActive(n);
    document.getElementById(`passage-${n}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  };

  const selectQuestion = (i: number) => {
    setIdx(i);
    setActive(null);
  };

  return (
    <div>
      <div className="mb-6">
        <div className="text-xs uppercase tracking-wider text-muted mb-2">example question</div>
        <div className="flex flex-col gap-1.5">
          {answers.map((q, i) => (
            <button
              key={q.id}
              onClick={() => selectQuestion(i)}
              className={
                "text-left text-sm px-3 py-2 rounded-md border transition-colors " +
                (i === idx
                  ? "border-accent bg-raised text-ink"
                  : "border-line text-muted hover:text-ink hover:border-muted")
              }
            >
              {q.question}
            </button>
          ))}
        </div>
      </div>

      <div className="grid lg:grid-cols-[1.1fr_0.9fr] gap-8">
        <section className="space-y-5">
          <h1 className="font-serif text-3xl leading-tight tracking-tight">{a.question}</h1>
          <DecisionBar a={a} />
          {a.abstained ? (
            <div className="border-l-2 pl-4 py-1" style={{ borderColor: "var(--warm)" }}>
              <p className="font-serif text-[1.28rem] leading-relaxed max-w-reading">
                The system declined to answer. The retrieved passages do not contain sufficient
                evidence, so it abstains rather than guess.
              </p>
              <p className="mt-2 text-sm text-muted">reason: {a.abstain_reason}</p>
            </div>
          ) : (
            <>
              <AnswerProse text={a.answer_text} onCite={onCite} active={active} />
              <p className="text-sm text-muted max-w-reading">{a.rationale}</p>
            </>
          )}
          <KvgateStrip a={a} />
        </section>

        <section>
          <div className="text-xs uppercase tracking-wider text-muted mb-3">
            retrieved evidence · top {a.contexts.length}
          </div>
          <div className="space-y-3">
            {a.contexts.map((c) => (
              <Passage key={c.marker} c={c} active={active === c.marker} />
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
