import type { Metadata } from "next";
import { Newsreader, Public_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const serif = Newsreader({
  subsets: ["latin"],
  variable: "--font-serif",
  display: "swap",
  style: ["normal", "italic"],
});
const sans = Public_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});
const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CiteMD - clinical evidence workstation",
  description:
    "A clinical-evidence RAG that answers with grounded citations and abstains when the evidence is weak. Research and evaluation only, not for clinical use.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${serif.variable} ${sans.variable} ${mono.variable}`}>{children}</body>
    </html>
  );
}
