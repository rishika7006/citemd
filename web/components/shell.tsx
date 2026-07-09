"use client";

import { useEffect, useState } from "react";
import type { AnswerRecord, EvalSummary, ServingSummary } from "@/lib/types";
import { Workstation } from "@/components/workstation";
import { EvaluationView } from "@/components/evaluation";
import { ServingView } from "@/components/serving";
import { DocsView } from "@/components/docs";
import { AboutSections } from "@/components/about";
import { RepoLink, ContactMenu } from "@/components/nav-actions";

type Tab = "workstation" | "serving" | "evaluation" | "docs";

const TABS: Tab[] = ["workstation", "serving", "evaluation", "docs"];

const TAB_LABEL: Record<Tab, string> = {
  workstation: "Workstation",
  serving: "Serving Path",
  evaluation: "Evaluation",
  docs: "Docs",
};

function ThemeToggle() {
  const [theme, setTheme] = useState<"light" | "dark" | null>(null);
  useEffect(() => {
    const saved = (localStorage.getItem("citemd-theme") as "light" | "dark") || null;
    if (saved) document.documentElement.setAttribute("data-theme", saved);
    setTheme(saved);
  }, []);
  const toggle = () => {
    const current =
      theme ??
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("citemd-theme", next);
    setTheme(next);
  };
  return (
    <button
      onClick={toggle}
      className="font-mono text-xs text-muted hover:text-ink transition-colors"
      aria-label="Toggle color theme"
    >
      {theme === "dark" ? "light" : "dark"}
    </button>
  );
}

export function Shell({
  answers,
  evalData,
  servingData,
}: {
  answers: AnswerRecord[];
  evalData: EvalSummary;
  servingData: ServingSummary;
}) {
  const [tab, setTab] = useState<Tab>("workstation");
  return (
    <div className="min-h-screen">
      <header className="border-b border-line">
        <div className="mx-auto max-w-6xl px-6 py-5 flex flex-wrap items-center justify-between gap-x-8 gap-y-4">
          <div className="flex items-baseline gap-4">
            <span className="font-serif text-2xl tracking-tight">CiteMD</span>
            <span className="hidden md:inline text-sm text-muted">
              Clinical Evidence Workstation
            </span>
          </div>
          <div className="flex items-center gap-5">
            <nav className="flex gap-4 text-sm">
              {TABS.map((t) => (
                <button
                  key={t}
                  onClick={() => setTab(t)}
                  className={
                    "pb-0.5 border-b-2 transition-colors " +
                    (tab === t
                      ? "border-accent text-ink"
                      : "border-transparent text-muted hover:text-ink")
                  }
                >
                  {TAB_LABEL[t]}
                </button>
              ))}
            </nav>
            <span className="hidden sm:block h-4 w-px bg-line" aria-hidden="true" />
            <div className="hidden sm:flex items-center gap-5">
              <RepoLink />
              <ContactMenu />
            </div>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        {tab === "workstation" && <Workstation answers={answers} />}
        {tab === "serving" && <ServingView data={servingData} />}
        {tab === "evaluation" && <EvaluationView data={evalData} />}
        {tab === "docs" && <DocsView />}
        {tab !== "docs" && <AboutSections />}
      </main>

      <footer className="border-t border-line mt-16">
        <div className="mx-auto max-w-6xl px-6 py-6 text-xs text-muted flex flex-wrap gap-x-8 gap-y-2 justify-between">
          <span>
            Research and evaluation only. Not for clinical use. Public, non-PHI data.
          </span>
          <span className="font-mono">
            demo mode: precomputed answers, no live model call
          </span>
        </div>
      </footer>
    </div>
  );
}
