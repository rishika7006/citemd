"""Fetch a small set of open-access clinical figures with captions for the multimodal
serving benchmark. Runs on the GPU pod (clean egress); uses the NLM Open-i API, which
returns biomedical figures from the PMC Open Access subset with captions.

    python benchmarks/fetch_clinical_figures.py --out images --n 48

Output: N image files in --out, plus captions.jsonl mapping {file, caption, source}.
Figures are public, open-access (PMC OA). Research and evaluation use only.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urljoin

import httpx

OPENI = "https://openi.nlm.nih.gov"
# Broad clinical imaging topics so the set resembles a real evidence corpus.
QUERIES = [
    "chest radiograph pneumonia",
    "brain MRI stroke",
    "abdominal CT",
    "histopathology carcinoma",
    "electrocardiogram",
    "fundus retinopathy",
    "bone fracture radiograph",
    "echocardiography",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="images")
    ap.add_argument("--n", type=int, default=48, help="Total figures to fetch.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per = max(1, args.n // len(QUERIES))
    manifest = []
    idx = 0
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for q in QUERIES:
            if idx >= args.n:
                break
            r = client.get(
                f"{OPENI}/api/search",
                params={"query": q, "it": "x,xg,c,m,mc,p,ph,u", "m": 1, "n": per},
            )
            r.raise_for_status()
            for item in (r.json().get("list") or [])[:per]:
                if idx >= args.n:
                    break
                img_rel = item.get("imgLarge") or item.get("imgThumb")
                caption = (item.get("image", {}) or {}).get("caption") or item.get("title") or q
                if not img_rel:
                    continue
                url = urljoin(OPENI, img_rel)
                try:
                    img = client.get(url)
                    img.raise_for_status()
                except httpx.HTTPError:
                    continue
                fn = f"fig_{idx:03d}.jpg"
                (out / fn).write_bytes(img.content)
                manifest.append({"file": fn, "caption": " ".join(caption.split())[:600],
                                  "source": urljoin(OPENI, item.get("detailedQueryURL", ""))})
                idx += 1
                time.sleep(0.2)
    (out / "captions.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    print(f"Fetched {idx} figures to {out}/ with captions.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
