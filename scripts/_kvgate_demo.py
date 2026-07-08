"""Capture a CiteMD -> KVGate -> Anthropic composition demo.

Sends the same request twice through a running KVGate (localhost:8080). The first call is a
real Anthropic call (cache miss); the second identical call is served from KVGate's response
cache (fast, no model call). Prints the KVGate metadata and timing for each. Not part of the
test suite; run with a KVGate instance running the `citemd` model.
"""

import time

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")
messages = [
    {"role": "system", "content": "You are a clinical-evidence assistant. Be concise."},
    {"role": "user", "content": "In one sentence, what is first-line pharmacotherapy for type 2 diabetes?"},
]


def call(label):
    t = time.perf_counter()
    resp = client.chat.completions.create(model="citemd", messages=messages)
    dt = (time.perf_counter() - t) * 1000
    kv = getattr(resp, "kvgate", None) or (resp.model_extra or {}).get("kvgate", {})
    text = resp.choices[0].message.content.strip()
    print(f"\n[{label}] wall={dt:.0f} ms")
    print(f"  kvgate={kv}")
    print(f"  answer={text[:120]}")
    return kv, dt


print("CiteMD OpenAI-compatible client -> KVGate (localhost:8080) -> backend")
call("call 1 (expected cache miss)")
call("call 2 (identical, expected cache hit)")
print("\nKVGATE_DEMO_OK")
