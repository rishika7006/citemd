import type { Config } from "tailwindcss";

// Palette and type are driven by CSS variables (see app/globals.css) so light and dark are
// deliberate, not an auto-invert. Tailwind only exposes them as named tokens.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "var(--paper)",
        raised: "var(--raised)",
        sunk: "var(--sunk)",
        ink: "var(--ink)",
        muted: "var(--muted)",
        faint: "var(--faint)",
        line: "var(--line)",
        accent: "var(--accent)",
        "accent-soft": "var(--accent-soft)",
        warm: "var(--warm)",
        good: "var(--good)",
        warn: "var(--warn)",
        bad: "var(--bad)",
      },
      fontFamily: {
        serif: "var(--font-serif)",
        sans: "var(--font-sans)",
        mono: "var(--font-mono)",
      },
      maxWidth: {
        reading: "68ch",
      },
    },
  },
  plugins: [],
};
export default config;
