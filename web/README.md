# CiteMD web

The Evidence Workstation: a front end for the CiteMD clinical-evidence RAG. It presents a
cited answer with its supporting passages, the abstention decision and confidence, the
retrieval mode, and the KVGate serving metadata, plus an evaluation view with the
risk-coverage curve, calibration, and the retrieval ablation.

Runs in demo mode by default: precomputed answers in `public/demo/`, no model call and no key.
The data shapes in `lib/types.ts` match the backend `CitedAnswer` and eval `summary.json`, so a
live `/ask` endpoint can replace the demo data without changing the components.

Stack: Next.js (App Router), TypeScript, Tailwind, hand-built SVG charts. Serif display type,
mono for identifiers and metrics, a single deep-teal accent, deliberate light and dark themes.

```bash
npm install
npm run dev     # http://localhost:3000
npm run build
```

Research and evaluation only. Not for clinical use.
