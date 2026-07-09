// Headless capture of the Evidence Workstation for the README and LinkedIn.
// Usage: node shot.mjs           (dev server on :3000)
//        PORT=50435 node shot.mjs
import puppeteer from "puppeteer";
import { mkdirSync } from "node:fs";

const OUT = "../docs";
const PORT = process.env.PORT || "3000";
mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({ headless: "new" });
const page = await browser.newPage();
await page.setViewport({ width: 1400, height: 1180, deviceScaleFactor: 2 });

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function setTheme(theme) {
  await page.evaluate((t) => {
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem("citemd-theme", t);
  }, theme);
  await wait(350);
}
async function tab(name) {
  await page.evaluate((n) => {
    const b = [...document.querySelectorAll("header nav button")].find(
      (x) => x.textContent.trim() === n,
    );
    if (b) b.click();
  }, name);
  await wait(450);
}
async function scrollToText(text) {
  await page.evaluate((t) => {
    const el = [...document.querySelectorAll("h1, h2, div")].find((x) =>
      x.textContent.trim().startsWith(t),
    );
    if (el) el.scrollIntoView({ block: "start" });
    else window.scrollTo(0, 0);
  }, text);
  await wait(300);
}

await page.goto(`http://localhost:${PORT}`, { waitUntil: "networkidle0" });
await wait(500);

for (const theme of ["dark", "light"]) {
  await setTheme(theme);

  await tab("Workstation");
  await page.evaluate(() => window.scrollTo(0, 0));
  await wait(200);
  await page.screenshot({ path: `${OUT}/workstation-${theme}.png` });

  await tab("Serving Path");
  await page.evaluate(() => window.scrollTo(0, 0));
  await wait(200);
  await page.screenshot({ path: `${OUT}/serving-${theme}.png` });

  await tab("Evaluation");
  await page.evaluate(() => window.scrollTo(0, 0));
  await wait(200);
  await page.screenshot({ path: `${OUT}/evaluation-${theme}.png` });

  console.log(`captured ${theme}`);
}

await browser.close();
console.log("SHOTS_DONE");
