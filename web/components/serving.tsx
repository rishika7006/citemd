import type { ServingSummary } from "@/lib/types";

function Bar({
  label,
  value,
  max,
  unit,
  color,
  note,
}: {
  label: string;
  value: number;
  max: number;
  unit: string;
  color: string;
  note?: string;
}) {
  const pct = Math.max(2, Math.min(100, (value / max) * 100));
  return (
    <div className="flex items-center gap-3 text-sm py-1">
      <div className="w-44 shrink-0 text-muted">{label}</div>
      <div className="flex-1 h-2 bg-sunk rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, background: color }} />
      </div>
      <div className="w-28 text-right font-mono tnum">
        {value.toLocaleString()}
        {unit}
        {note ? <span className="text-muted"> {note}</span> : null}
      </div>
    </div>
  );
}

function Section({
  title,
  desc,
  children,
}: {
  title: string;
  desc: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-12">
      <h2 className="font-serif text-xl mb-1">{title}</h2>
      <p className="text-sm text-muted max-w-3xl mb-5">{desc}</p>
      {children}
    </section>
  );
}

const GREY = "var(--muted)";
const TEAL = "var(--accent)";
const WARM = "var(--warm)";

export function ServingView({ data }: { data: ServingSummary }) {
  const r = data.routing;
  const lm = data.lmcache;
  const routeMaxT = Math.max(r.round_robin.ttft_p95, r.prefix_aware.ttft_p95);
  const lmMaxT = Math.max(...lm.arms.map((a) => a.ttft_p95));
  const lmMaxThr = Math.max(...lm.arms.map((a) => a.thr));

  return (
    <div>
      <p className="text-sm text-muted max-w-3xl mb-8">
        CiteMD talks to a model through one OpenAI-compatible client, so it can run direct to a
        provider, through KVGate to a provider, or through KVGate to self-hosted vLLM with
        LMCache. This page reports what KVGate and LMCache add, measured on a real multimodal
        clinical workload. It is a serving-layer benchmark (latency, throughput, cache), not a
        medical-accuracy benchmark.
      </p>

      <div className="font-mono text-xs text-muted mb-10 flex flex-wrap gap-x-2 gap-y-1">
        <span className="text-ink">CiteMD</span> <span>-&gt;</span>
        <span className="text-ink">KVGate</span> <span>-&gt;</span>
        <span>hosted API (response cache, rate limit, metrics)</span>
        <span className="mx-1">|</span>
        <span>or self-hosted vLLM + LMCache (KV routing, offload)</span>
      </div>

      <Section
        title="Response cache hit"
        desc="Through KVGate, an identical repeat request is served from the response cache with no model call. Measured on the hosted path (CPU only, no GPU)."
      >
        <Bar label="first call (cache miss)" value={data.cache_hit.miss_ms} max={data.cache_hit.miss_ms} unit=" ms" color={WARM} />
        <Bar label="repeat call (cache hit)" value={data.cache_hit.hit_ms} max={data.cache_hit.miss_ms} unit=" ms" color={TEAL} note="no model call" />
      </Section>

      <Section
        title="KVGate prefix-aware routing (2 replicas)"
        desc={`Two vLLM replicas behind KVGate, same multimodal trace under round-robin versus prefix-aware routing. Prefix-aware sends requests that share a prefix to the replica holding its KV warm: TTFT p50 ${r.ttft_p50_delta}%, p95 ${r.ttft_p95_delta}%, throughput +${r.thr_delta}%, at ${(r.prefix_aware.affinity * 100).toFixed(1)}% affinity.`}
      >
        <div className="grid sm:grid-cols-2 gap-x-10 gap-y-6">
          <div>
            <div className="text-xs uppercase tracking-wider text-muted mb-2">TTFT p50, lower better</div>
            <Bar label="round-robin" value={r.round_robin.ttft_p50} max={routeMaxT} unit=" ms" color={GREY} />
            <Bar label="prefix-aware (KVGate)" value={r.prefix_aware.ttft_p50} max={routeMaxT} unit=" ms" color={TEAL} />
            <div className="text-xs uppercase tracking-wider text-muted mt-4 mb-2">TTFT p95, lower better</div>
            <Bar label="round-robin" value={r.round_robin.ttft_p95} max={routeMaxT} unit=" ms" color={GREY} />
            <Bar label="prefix-aware (KVGate)" value={r.prefix_aware.ttft_p95} max={routeMaxT} unit=" ms" color={TEAL} />
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-muted mb-2">Throughput, higher better</div>
            <Bar label="round-robin" value={r.round_robin.thr} max={r.prefix_aware.thr} unit=" rps" color={GREY} />
            <Bar label="prefix-aware (KVGate)" value={r.prefix_aware.thr} max={r.prefix_aware.thr} unit=" rps" color={WARM} />
          </div>
        </div>
      </Section>

      <Section
        title="LMCache under GPU memory pressure"
        desc={`Single replica, GPU KV capped so the working set overflows. CPU L1 offload cut TTFT p50 ${lm.ttft_p50_delta}%; adding Redis L2 cut tail latency p95 ${lm.ttft_p95_delta}% and raised throughput +${lm.thr_delta}%.`}
      >
        <div className="grid sm:grid-cols-2 gap-x-10 gap-y-6">
          <div>
            <div className="text-xs uppercase tracking-wider text-muted mb-2">TTFT p95, lower better</div>
            {lm.arms.map((a) => (
              <Bar key={a.name} label={a.name} value={a.ttft_p95} max={lmMaxT} unit=" ms"
                color={a.name.startsWith("baseline") ? GREY : TEAL}
                note={a.hit != null ? `${a.hit}% hit` : undefined} />
            ))}
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-muted mb-2">Throughput, higher better</div>
            {lm.arms.map((a) => (
              <Bar key={a.name} label={a.name} value={a.thr} max={lmMaxThr} unit=" rps"
                color={a.name.startsWith("baseline") ? GREY : WARM} />
            ))}
          </div>
        </div>
      </Section>

      <Section
        title="The honest crossover: no pressure, no gain"
        desc="When the working set fits in GPU memory, LMCache is net overhead. vLLM's own in-GPU prefix cache already serves the KV, so the extra offload path only adds cost. Enable LMCache by the working-set-to-GPU ratio, not by default."
      >
        <div className="grid sm:grid-cols-2 gap-x-10 gap-y-2 text-sm">
          <div>
            <div className="text-xs uppercase tracking-wider text-muted mb-2">Fits GPU (no pressure), TTFT p50</div>
            <Bar label="baseline" value={data.crossover.no_pressure.baseline} max={400} unit=" ms" color={TEAL} />
            <Bar label="+ L1" value={data.crossover.no_pressure.l1} max={400} unit=" ms" color={GREY} />
            <Bar label="+ L1 + L2" value={data.crossover.no_pressure.l1l2} max={400} unit=" ms" color={GREY} />
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-muted mb-2">Overflows GPU (pressure), TTFT p50</div>
            <Bar label="baseline" value={data.crossover.pressure.baseline} max={400} unit=" ms" color={GREY} />
            <Bar label="+ L1" value={data.crossover.pressure.l1} max={400} unit=" ms" color={TEAL} />
            <Bar label="+ L1 + L2" value={data.crossover.pressure.l1l2} max={400} unit=" ms" color={TEAL} />
          </div>
        </div>
      </Section>

      <div className="border-t border-line pt-6 text-sm text-muted max-w-3xl space-y-2">
        <p>
          <span className="text-ink">KVGate response cache</span> helps repeated requests in any
          mode, hosted or self-hosted, by skipping the model.
        </p>
        <p>
          <span className="text-ink">LMCache</span> applies only to self-hosted vLLM on GPU, and
          only helps when the KV overflows GPU memory. Prefix-aware routing and LMCache are
          different mechanisms and are reported separately.
        </p>
        <p className="font-mono text-xs">
          {data.hardware} · {data.stack} · {data.workload}
        </p>
      </div>
    </div>
  );
}
