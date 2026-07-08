// Headless capture of the Evidence Workstation for the README and LinkedIn.
// Usage: node shot.mjs  (dev server must be running on :3000)
import puppeteer from "puppeteer";
import { mkdirSync } from "node:fs";

const OUT = "../docs";
mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({ headless: "new" });
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1180, deviceScaleFactor: 2 });

async function setTheme(theme) {
  await page.evaluate((t) => {
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem("citemd-theme", t);
  }, theme);
  await new Promise((r) => setTimeout(r, 350));
}
async function tab(name) {
  await page.evaluate((n) => {
    const b = [...document.querySelectorAll("nav button")].find((x) => x.textContent.trim() === n);
    if (b) b.click();
  }, name);
  await new Promise((r) => setTimeout(r, 450));
}

await page.goto("http://localhost:3000", { waitUntil: "networkidle0" });
await new Promise((r) => setTimeout(r, 500));

for (const theme of ["dark", "light"]) {
  await setTheme(theme);
  await tab("Workstation");
  await page.screenshot({ path: `${OUT}/workstation-${theme}.png` });
  await tab("Evaluation");
  await page.screenshot({ path: `${OUT}/evaluation-${theme}.png` });
  console.log(`captured ${theme}`);
}

await browser.close();
console.log("SHOTS_DONE");
