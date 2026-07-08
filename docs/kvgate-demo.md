# CiteMD through KVGate: captured composition demo

This shows CiteMD's OpenAI-compatible client making generation requests through a running
[KVGate](https://github.com/rishika7006/kvgate) instance, and KVGate's response cache turning an
identical repeat request from a real call into a near-instant cache hit. It uses KVGate's
built-in mock backend, so it needs no API key and spends nothing; the mock proves the serving
path and caching. Pointing at a real provider is one config block (see below).

## Run it

```bash
# 1. KVGate (in the kvgate repo), mock backend, response cache on:
kvgate run -c config/citemd-demo.yaml        # serves the "citemd" model on :8080

# 2. From CiteMD, call it twice through the OpenAI-compatible client:
python scripts/_kvgate_demo.py
```

## Captured output

```
CiteMD OpenAI-compatible client -> KVGate (localhost:8080) -> backend

[call 1 (expected cache miss)] wall=561 ms
  kvgate={'cache': 'miss', 'provider': 'mock-fast', 'upstream_model': 'mock-fast',
          'upstream_latency_ms': 121.31, 'latency_ms': 127.27, 'estimated_cost_usd': 0.000374}

[call 2 (identical, expected cache hit)] wall=2 ms
  kvgate={'cache': 'exact', 'similarity': 1.0, 'latency_ms': 0.13}
```

The first call is a cache miss and is served by the backend. The identical second call is an
exact cache hit: KVGate returns it in about 2 ms with no backend call and no cost. Each response
carries KVGate metadata (cache status, provider, latency, cost) that the CiteMD front end
displays in its serving strip.

## Pointing at a real provider

Swap the provider and model blocks in the KVGate config, keeping everything else:

```yaml
providers:
  - name: anthropic
    type: anthropic
    api_key: ${ANTHROPIC_API_KEY:-}
models:
  - name: citemd
    deployments:
      - provider: anthropic
        model: <an Anthropic model id>
```

CiteMD does not change: it already points at `CITEMD_LLM_BASE_URL=http://localhost:8080/v1` with
`CITEMD_LLM_MODEL=citemd`. Note: KVGate's Anthropic provider currently sends both `temperature`
and `top_p`, which some newer Anthropic models reject; that is a KVGate-side fix, tracked in the
KVGate repo, and does not affect the mock demo above or the CiteMD integration.
