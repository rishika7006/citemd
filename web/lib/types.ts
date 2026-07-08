// Shapes shared by the demo data and (later) the live /ask API. Kept deliberately close to
// the backend CitedAnswer and eval summary so a real endpoint can drop in unchanged.

export interface Citation {
  marker: number;
  source_id: string;
  title: string;
  section: string | null;
  score: number;
  uri: string;
}

export interface Context {
  marker: number;
  source_id: string;
  title: string;
  section: string | null;
  score: number;
  text: string;
  cited: boolean;
  // Exact substring of `text` the model relied on, highlighted in the reader.
  support: string | null;
}

export interface KvgateMeta {
  enabled: boolean;
  cache: "miss" | "exact" | "semantic";
  provider: string;
  model: string;
  latency_ms: number;
  cost_usd: number;
}

export interface AnswerRecord {
  id: string;
  dataset: string;
  question: string;
  options: Record<string, string>;
  gold: string;
  predicted: string | null;
  abstained: boolean;
  abstain_reason: string;
  confidence: number;
  retrieval_mode: string;
  // Answer prose with inline [n] markers referring to citations.
  answer_text: string;
  rationale: string;
  citations: Citation[];
  contexts: Context[];
  gold_retrieved: boolean;
  gold_rank: number | null;
  kvgate: KvgateMeta;
}

export interface Arm {
  slug: string;
  name: string;
  accuracy: number;
  acc_lo: number;
  acc_hi: number;
  error: number;
  hit_rate: number | null;
  ece: number;
}

export interface CoveragePoint {
  coverage: number;
  accuracy: number;
  risk: number;
}

export interface CalibrationBin {
  confidence: number;
  accuracy: number;
  n: number;
}

export interface EvalSummary {
  dataset: string;
  n: number;
  seed: number;
  model: string;
  best_arm: string;
  arms: Arm[];
  abstention: {
    fraction: number;
    base_error: number;
    kept_error: number;
    rel_reduction: number;
    coverage: number;
  };
  ece: number;
  risk_coverage: CoveragePoint[];
  calibration: CalibrationBin[];
  commands: string[];
}
