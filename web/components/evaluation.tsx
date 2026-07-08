"use client";

import { useMemo, useState } from "react";
import type { CalibrationBin, CoveragePoint, EvalSummary } from "@/lib/types";

function pctText(x: number, digits = 1) {
  return `${(x * 100).toFixed(digits)}%`;
}

function AblationTable({ data }: { data: EvalSummary }) {
  const maxAcc = 1;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="text-left text-xs uppercase tracking-wider text-muted">
            <th className="py-2 pr-4 font-normal">configuration</th>
            <th className="py-2 pr-4 font-normal">accuracy (95% CI)</th>
            <th className="py-2 pr-4 font-normal">retrieval hit-rate</th>
            <th className="py-2 pr-4 font-normal">ECE</th>
            <th className="py-2 font-normal w-40">accuracy</th>
          </tr>
        </thead>
        <tbody>
          {data.arms.map((arm) => (
            <tr key={arm.slug} className="border-t border-line">
              <td className="py-2.5 pr-4">{arm.name}</td>
              <td className="py-2.5 pr-4 font-mono tnum">
                {pctText(arm.accuracy)}
                <span className="text-muted">
                  {" "}
                  ({pctText(arm.acc_lo, 0)}-{pctText(arm.acc_hi, 0)})
                </span>
              </td>
              <td className="py-2.5 pr-4 font-mono tnum text-muted">
                {arm.hit_rate === null ? "-" : pctText(arm.hit_rate)}
              </td>
              <td className="py-2.5 pr-4 font-mono tnum text-muted">{arm.ece.toFixed(3)}</td>
              <td className="py-2.5">
                <div className="h-1.5 bg-sunk rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${(arm.accuracy / maxAcc) * 100}%`, background: "var(--accent)" }}
                  />
                </div>
              </td>
            </tr>
          ))}
          <tr className="border-t-2 border-ink/30">
            <td className="py-2.5 pr-4">
              {data.arms.find((x) => x.slug === data.best_arm)?.name ?? "best"} + abstention @{" "}
              {pctText(data.abstention.fraction, 0)}
            </td>
            <td className="py-2.5 pr-4 font-mono tnum">
              {pctText(1 - data.abstention.kept_error)}
              <span className="text-muted"> · {pctText(data.abstention.coverage, 0)} coverage</span>
            </td>
            <td className="py-2.5 pr-4 text-muted">-</td>
            <td className="py-2.5 pr-4 text-muted">-</td>
            <td className="py-2.5">
              <div className="h-1.5 bg-sunk rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${(1 - data.abstention.kept_error) * 100}%`,
                    background: "var(--warm)",
                  }}
                />
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

const W = 620;
const H = 320;
const PAD = { l: 52, r: 18, t: 18, b: 40 };
const px = (v: number) => PAD.l + v * (W - PAD.l - PAD.r);
const py = (v: number, max: number) => H - PAD.b - (v / max) * (H - PAD.t - PAD.b);

function RiskCoverage({ points }: { points: CoveragePoint[] }) {
  const sorted = useMemo(() => [...points].sort((a, b) => a.coverage - b.coverage), [points]);
  const maxRisk = Math.max(0.05, ...sorted.map((p) => p.risk)) * 1.1;
  const [coverage, setCoverage] = useState(0.8);

  const path = sorted
    .map((p, i) => `${i === 0 ? "M" : "L"} ${px(p.coverage).toFixed(1)} ${py(p.risk, maxRisk).toFixed(1)}`)
    .join(" ");

  const op = sorted.reduce((prev, p) =>
    Math.abs(p.coverage - coverage) < Math.abs(prev.coverage - coverage) ? p : prev,
  );
  const yTicks = [0, maxRisk / 2, maxRisk];

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Risk-coverage curve">
        {yTicks.map((t, i) => (
          <g key={i}>
            <line
              x1={PAD.l}
              x2={W - PAD.r}
              y1={py(t, maxRisk)}
              y2={py(t, maxRisk)}
              stroke="var(--line)"
              strokeWidth={1}
            />
            <text x={PAD.l - 8} y={py(t, maxRisk) + 4} textAnchor="end" className="fill-muted" style={{ fontSize: 11 }}>
              {pctText(t, 0)}
            </text>
          </g>
        ))}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <text key={t} x={px(t)} y={H - 14} textAnchor="middle" className="fill-muted" style={{ fontSize: 11 }}>
            {pctText(t, 0)}
          </text>
        ))}
        <line x1={px(coverage)} x2={px(coverage)} y1={PAD.t} y2={H - PAD.b} stroke="var(--warm)" strokeWidth={1} strokeDasharray="3 3" />
        <path d={path} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinejoin="round" />
        {sorted.map((p, i) => (
          <circle key={i} cx={px(p.coverage)} cy={py(p.risk, maxRisk)} r={2.5} fill="var(--accent)" />
        ))}
        <circle cx={px(op.coverage)} cy={py(op.risk, maxRisk)} r={5} fill="var(--warm)" stroke="var(--paper)" strokeWidth={2} />
        <text x={W - PAD.r} y={PAD.t + 10} textAnchor="end" className="fill-muted" style={{ fontSize: 11 }}>
          error among answered
        </text>
      </svg>

      <div className="mt-3 flex flex-col sm:flex-row sm:items-center gap-3">
        <input
          type="range"
          min={Math.min(...sorted.map((p) => p.coverage))}
          max={1}
          step={0.01}
          value={coverage}
          onChange={(e) => setCoverage(Number(e.target.value))}
          className="flex-1 accent-[var(--warm)]"
          aria-label="coverage"
        />
        <div className="font-mono text-sm tnum whitespace-nowrap">
          coverage <span className="text-ink">{pctText(op.coverage, 0)}</span>
          <span className="text-muted"> · </span>
          error <span className="text-ink">{pctText(op.risk)}</span>
          <span className="text-muted"> · abstain {pctText(1 - op.coverage, 0)}</span>
        </div>
      </div>
    </div>
  );
}

function Calibration({ bins, ece }: { bins: CalibrationBin[]; ece: number }) {
  const cw = 300;
  const ch = 300;
  const p = 40;
  const sx = (v: number) => p + v * (cw - p - 14);
  const sy = (v: number) => ch - p - v * (ch - p - 14);
  const pts = [...bins].sort((a, b) => a.confidence - b.confidence);
  const line = pts.map((b, i) => `${i === 0 ? "M" : "L"} ${sx(b.confidence).toFixed(1)} ${sy(b.accuracy).toFixed(1)}`).join(" ");
  return (
    <div>
      <svg viewBox={`0 0 ${cw} ${ch}`} className="w-full max-w-[320px]" role="img" aria-label="Reliability diagram">
        <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="var(--faint)" strokeWidth={1} strokeDasharray="4 4" />
        {[0, 0.5, 1].map((t) => (
          <g key={t}>
            <text x={sx(t)} y={ch - 14} textAnchor="middle" className="fill-muted" style={{ fontSize: 11 }}>
              {pctText(t, 0)}
            </text>
            <text x={p - 8} y={sy(t) + 4} textAnchor="end" className="fill-muted" style={{ fontSize: 11 }}>
              {pctText(t, 0)}
            </text>
          </g>
        ))}
        <path d={line} fill="none" stroke="var(--accent)" strokeWidth={2} />
        {pts.map((b, i) => (
          <circle key={i} cx={sx(b.confidence)} cy={sy(b.accuracy)} r={4} fill="var(--accent)" />
        ))}
        <text x={sx(1)} y={sy(1) - 8} textAnchor="end" className="fill-muted" style={{ fontSize: 11 }}>
          perfect
        </text>
      </svg>
      <div className="mt-1 text-sm text-muted">
        ECE <span className="font-mono text-ink tnum">{ece.toFixed(3)}</span>
        <span className="text-muted"> · lower is better. Confidence vs empirical accuracy.</span>
      </div>
    </div>
  );
}

function ReproPanel({ data }: { data: EvalSummary }) {
  return (
    <div className="border border-line rounded-md bg-sunk">
      <div className="px-4 py-2 border-b border-line flex items-center justify-between">
        <span className="text-xs uppercase tracking-wider text-muted">reproduce</span>
        <span className="font-mono text-xs text-muted">
          {data.dataset} · n={data.n} · seed={data.seed} · {data.model}
        </span>
      </div>
      <pre className="px-4 py-3 font-mono text-xs leading-relaxed overflow-x-auto text-ink">
        {data.commands.join("\n")}
      </pre>
    </div>
  );
}

export function EvaluationView({ data }: { data: EvalSummary }) {
  return (
    <div className="space-y-10">
      <section>
        <h2 className="font-serif text-2xl mb-1">Selective prediction</h2>
        <p className="text-sm text-muted max-w-reading mb-5">
          Abstaining on the least-confident questions trades a little coverage for lower error on
          the answered set. Drag the coverage control to see the operating point. At{" "}
          {pctText(data.abstention.coverage, 0)} coverage, error falls from{" "}
          {pctText(data.abstention.base_error)} to {pctText(data.abstention.kept_error)}, a{" "}
          {pctText(data.abstention.rel_reduction, 0)} relative reduction.
        </p>
        <div className="grid lg:grid-cols-[1.4fr_1fr] gap-8 items-start">
          <div className="border border-line rounded-md bg-raised p-4">
            <RiskCoverage points={data.risk_coverage} />
          </div>
          <div className="border border-line rounded-md bg-raised p-4">
            <div className="text-xs uppercase tracking-wider text-muted mb-3">calibration</div>
            <Calibration bins={data.calibration} ece={data.ece} />
          </div>
        </div>
      </section>

      <section>
        <h2 className="font-serif text-2xl mb-1">Retrieval ablation</h2>
        <p className="text-sm text-muted max-w-reading mb-5">
          Vector-only, hybrid, and hybrid with reranking, then the best arm with abstention.
          Reported with confidence intervals. On this dataset the retrieval variants do not
          separate beyond sampling noise; the gold source is already retrieved in nearly every
          case, so the reader and the abstention gate are where the movement is.
        </p>
        <AblationTable data={data} />
      </section>

      <section>
        <h2 className="font-serif text-2xl mb-3">Reproducibility</h2>
        <ReproPanel data={data} />
      </section>
    </div>
  );
}
