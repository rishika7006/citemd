# Mode B multimodal serving benchmark (CiteMD through KVGate to vLLM + LMCache MP)

Goal: show what KVGate and LMCache add for a real multimodal clinical workload, on GPU. This
reuses the proven MP-mode harness from KVGate PR #1 (`lmcache-mp-l2-analysis`) and supplies real
open-access clinical figures instead of synthetic images. It measures the serving layer
(latency, throughput, KV cache behavior), not medical accuracy.

## Hardware
- 2x A40 (48 GB) on RunPod. One replica per GPU for the routing arm; the LMCache MP cache-layer
  arms use a single replica.
- Runtime about 4.5 to 5.5 hours including setup. Estimated GPU cost about $4 to $6, budget $12.

## 0. Pod setup (run once, in tmux so it survives SSH drops)
```bash
tmux new -s bench
git clone -b lmcache-mp-l2-analysis https://github.com/rishika7006/kvgate.git && cd kvgate
python3 -m venv ~/venv && source ~/venv/bin/activate && pip install -U pip
pip install -r benchmarks/requirements-mp.txt      # vllm 0.23.0, lmcache 0.5.0
pip install -e ".[dev]"                              # KVGate gateway + load client
redis-server --daemonize yes

# CiteMD overlay: real clinical figures instead of synthetic images
curl -sSL https://raw.githubusercontent.com/rishika7006/citemd/main/benchmarks/fetch_clinical_figures.py -o fetch_figs.py
pip install httpx
python fetch_figs.py --out images --n 48           # PMC Open Access figures + captions.jsonl
export MODEL=llava-hf/llava-onevision-qwen2-7b-ov-hf
```

## 1. Cache layer: LMCache MP (CPU L1 + Redis L2), single replica
The KVGate harness sweeps the working set (8/24/48 figures) across baseline / L1 / L1+L2, cold
then warm x3, plus a persistence test (kill vLLM and the MP server, keep Redis).
```bash
tmux new -d -s sweep "bash benchmarks/lmcache_mp/run.sh > ~/mpout/sweep.log 2>&1"
# Arms captured: baseline (GPU only), L1=CPU, L1+Redis L2. Metrics scraped around each warm phase.
```

## 2. Routing: KVGate prefix-aware across two replicas
Bring up one vLLM+LMCache(MP) replica per GPU on 18001/18002, front them with KVGate.
```bash
# config/gpu.runpod.yaml already defines r1/r2 and the vlm model.
kvgate run -c config/gpu.runpod.yaml --port 8080 &     # routing.strategy: round_robin (arm), then prefix_kv_aware
python loadtest/multimodal_bench.py --host http://localhost:8080 --model vlm \
  --images-dir images --images 12 --sessions 120 --turns 1 --concurrency 8 --out D.json   # round_robin
# flip routing.strategy to prefix_kv_aware, restart gateway, rerun to E.json
python scripts/compare_results.py D=D.json E=E.json
```

## 3. Arms recorded for CiteMD
1. Direct vLLM (no KVGate, no LMCache): baseline TTFT/throughput.
2. KVGate to vLLM, no LMCache: gateway overhead (expect ~1 ms).
3. KVGate to vLLM + LMCache CPU L1: offload latency win under pressure.
4. KVGate to vLLM + LMCache CPU L1 + Redis L2 (MP): persistence and capacity; survives restart.
5. KVGate prefix-aware routing across 2 replicas: affinity, TTFT, throughput.
6. KVGate response-cache hit for a repeated CiteMD question (already captured, CPU-only).

## 4. Metrics
TTFT p50/p95/p99, total latency percentiles, inter-token latency, throughput, req/s, tokens/s,
GPU memory, CPU L1 bytes, Redis L2 keys and used_memory, KV hit/miss/reload, KVGate routing
metadata and response-cache hit/miss, provider path, run cost. Sources: vLLM `/metrics`, LMCache
MP metrics, KVGate `/metrics` and `/admin/stats`, and the load client JSON.

## 5. Pull results back
```bash
# On the pod, push the result JSON + logs to a branch or scp them off:
mkdir -p out && cp -r ~/mpout D.json E.json out/ && tar czf citemd_modeb_results.tgz out
# scp citemd_modeb_results.tgz to the workstation; charts + writeup are rendered locally.
```

## Honest reporting
- LMCache helps only under GPU memory pressure (working set overflows GPU KV). When it fits, the
  cache is net overhead; report the crossover, do not hide it.
- MP mode is not faster per request; its value is persistence across restarts and a shared L2.
- The base vision model is general, so clinical answer quality is not the point here; the serving
  metrics are. CiteMD's app answers use hosted Claude vision, reported separately.
