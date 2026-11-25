#!/usr/bin/env python3
"""Steady-state decode & cold TTFT benchmark harness for Inkling-2-Mini (OpenAI-compatible vLLM server).

Talks to a live OpenAI-compatible vLLM server (default http://127.0.0.1:8000):
  1. Coherence & greedy correctness sanity check (3 prompts)
  2. Pure steady-state decode throughput & ITL across concurrencies C = 1, 4, 8, 16, 32
     at target context length (primes a shared prefix + unique suffix so all C requests
     finish prefill immediately and decode simultaneously in pure decode batches, measuring
     strictly over the all-C-active token timestamp window).
  3. Cold TTFT across prompt lengths 128, 512, 1024, 4096, 8192 tokens (unique token-0
     prefixes so prefix cache hit rate is strictly 0%).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import urllib.request
from statistics import median


_VOCAB = [
    "kernel", "volta", "tensor", "matrix", "stream", "warp", "memory", "cache",
    "block", "shared", "register", "pipeline", "attention", "expert", "router",
    "residual", "norm", "softmax", "logit", "token", "layer", "graph", "nvlink",
    "quant", "scale", "vector", "stride", "offset", "buffer", "device", "thread",
    "latency",
]


def _make_prompt_words(approx_tokens: int, seed: int = 0) -> str:
    n_words = max(16, int(approx_tokens * 0.88))
    out = [f"Request {seed} technical analysis:"]
    for i in range(n_words):
        out.append(_VOCAB[(i * 7 + seed * 13) % len(_VOCAB)])
    out.append("Summarize the key architectural points concisely:")
    return " ".join(out)


def _make_shared_ctx_prompt(ctx_tokens: int, req_id: int, prefix_seed: int) -> str:
    # Shared prefix of ~(ctx_tokens - 32) words followed by a unique 24-word suffix so
    # all C requests share the cached prefix prefill and enter pure concurrent decode together.
    n_shared = max(64, int((ctx_tokens - 32) * 0.88))
    shared = [f"SharedDocument {prefix_seed}:"]
    for i in range(n_shared):
        shared.append(_VOCAB[(i * 7 + prefix_seed * 11) % len(_VOCAB)])
    suffix = [f"QueryVariant {req_id}:"]
    for i in range(24):
        suffix.append(_VOCAB[(i * 5 + req_id * 17) % len(_VOCAB)])
    suffix.append("Answer concisely:")
    return " ".join(shared) + " " + " ".join(suffix)


def _post_stream_sync(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    tok_times: list[float] = []
    chunks: list[str] = []
    usage = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("text")
                if delta is None:
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                if delta:
                    tok_times.append(time.perf_counter())
                    chunks.append(delta)
    t1 = time.perf_counter()
    if not tok_times:
        raise RuntimeError(f"Empty SSE stream from {url} (usage={usage})")
    ttft_ms = (tok_times[0] - t0) * 1000.0
    gen_ms = (tok_times[-1] - tok_times[0]) * 1000.0
    comp_toks = (usage or {}).get("completion_tokens", len(tok_times))
    prompt_toks = (usage or {}).get("prompt_tokens", 0)
    return {
        "text": "".join(chunks),
        "ttft_ms": ttft_ms,
        "gen_ms": gen_ms,
        "tok_times": tok_times,
        "prompt_tokens": prompt_toks,
        "completion_tokens": comp_toks,
        "itl_ms": gen_ms / max(len(tok_times) - 1, 1),
    }


async def _run_concurrent_decode(
    url: str,
    model: str,
    conc: int,
    ctx_tokens: int,
    max_tokens: int,
    prefix_seed: int,
) -> dict:
    loop = asyncio.get_running_loop()
    # Prime shared KV prefix once so concurrent requests hit prefix cache and start decoding together.
    await loop.run_in_executor(
        None,
        _post_stream_sync,
        url,
        {
            "model": model,
            "prompt": _make_shared_ctx_prompt(ctx_tokens, req_id=9999, prefix_seed=prefix_seed),
            "max_tokens": 2,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )

    async def one(idx: int) -> dict:
        payload = {
            "model": model,
            "prompt": _make_shared_ctx_prompt(ctx_tokens, req_id=idx, prefix_seed=prefix_seed),
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        return await loop.run_in_executor(None, _post_stream_sync, url, payload)

    results = await asyncio.gather(*(one(i) for i in range(conc)))
    tot_prompt = sum(r["prompt_tokens"] for r in results)
    med_ttft = median(r["ttft_ms"] for r in results)

    # Measure strictly inside the pure-decode window where ALL C requests are concurrently decoding:
    # start after the last request's 3rd SSE chunk (0 prefill interference), stop at first request's last chunk.
    # NOTE: Under speculative decoding (DSpark), one SSE chunk carries 1..K+1 accepted tokens per step.
    # Scale chunk counts inside the overlap window by each request's exact (completion_tokens / num_sse_chunks)
    # ratio so both target-only (1 tok/chunk) and speculative decoding report true token throughput!
    t_start = max(r["tok_times"][min(2, len(r["tok_times"]) - 1)] for r in results)
    t_end = min(r["tok_times"][-1] for r in results)
    if t_end > t_start + 0.05:
        win_s = t_end - t_start
        win_chunks = 0
        win_toks_f = 0.0
        for r in results:
            c_in = sum(1 for t in r["tok_times"] if t_start < t <= t_end)
            win_chunks += c_in
            toks_per_chunk = r["completion_tokens"] / max(len(r["tok_times"]), 1)
            win_toks_f += c_in * toks_per_chunk
        win_toks = int(round(win_toks_f))
        agg_dec_tps = win_toks_f / win_s
        per_req_tps = agg_dec_tps / conc
        steady_itl_ms = 1000.0 / max(per_req_tps, 1e-6)
        step_ms = (win_s * 1000.0) / max(win_chunks / conc, 1e-6)
        avg_tpc = win_toks_f / max(win_chunks, 1)
    else:
        med_gen_s = median(r["gen_ms"] for r in results) / 1000.0
        med_comp = median(r["completion_tokens"] for r in results)
        med_chunks = median(len(r["tok_times"]) for r in results)
        avg_tpc = med_comp / max(med_chunks, 1)
        per_req_tps = max(med_comp - avg_tpc, 1.0) / max(med_gen_s, 1e-6)
        agg_dec_tps = per_req_tps * conc
        steady_itl_ms = 1000.0 / max(per_req_tps, 1e-6)
        step_ms = (med_gen_s * 1000.0) / max(med_chunks - 1, 1)
        win_toks = sum(r["completion_tokens"] for r in results)
        win_chunks = sum(len(r["tok_times"]) for r in results)

    return {
        "conc": conc,
        "prompt_tokens_avg": tot_prompt / conc,
        "med_ttft_ms": med_ttft,
        "steady_itl_ms": steady_itl_ms,
        "step_ms": step_ms,
        "tok_per_step": avg_tpc,
        "per_req_tps": per_req_tps,
        "agg_dec_tps": agg_dec_tps,
        "win_toks": win_toks,
        "win_chunks": win_chunks,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--mode", choices=["sanity", "decode", "ttft", "all"], default="all")
    ap.add_argument("--concs", default="1,4,8,16,32")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--max-tokens", type=int, default=96)
    args = ap.parse_args()
    url = f"{args.base_url}/v1/completions"

    if args.mode in ("sanity", "all"):
        print("=== 1. Coherence & Greedy Sanity Check ===", flush=True)
        for p in [
            "The capital of France is",
            "def fibonacci(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
            "Explain in two sentences why Volta V100 GPUs use FP16 Tensor Cores with FP32 accumulation:",
        ]:
            r = _post_stream_sync(
                url,
                {
                    "model": args.model,
                    "prompt": p,
                    "max_tokens": 48,
                    "temperature": 0.0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
            print(f"  PROMPT: {p!r}")
            print(
                f"  OUTPUT ({r['completion_tokens']} tok, TTFT={r['ttft_ms']:.1f}ms, "
                f"ITL={r['itl_ms']:.2f}ms): {r['text']!r}\n",
                flush=True,
            )

    if args.mode in ("decode", "all"):
        print(f"=== 2. Pure Steady-State Decode (ctx~{args.ctx}, max_tokens={args.max_tokens}) ===", flush=True)
        for idx, c in enumerate(int(x) for x in args.concs.split(",")):
            res = asyncio.run(
                _run_concurrent_decode(
                    url, args.model, c, args.ctx, args.max_tokens, prefix_seed=2000 + idx
                )
            )
            print(
                f"  C={c:2d} | p_tok={res['prompt_tokens_avg']:6.0f} | "
                f"TTFT={res['med_ttft_ms']:6.1f} ms | ITL={res['steady_itl_ms']:6.2f} ms | "
                f"step={res['step_ms']:5.2f} ms ({res['tok_per_step']:.2f} tok/step) | "
                f"user={res['per_req_tps']:6.1f} tok/s | agg={res['agg_dec_tps']:7.1f} tok/s "
                f"(win_toks={res['win_toks']})",
                flush=True,
            )

    if args.mode in ("ttft", "all"):
        print("\n=== 3. Cold TTFT Sweep (C=1, unique prompts, 0% prefix cache) ===", flush=True)
        # Warm up prefill kernels once
        _post_stream_sync(
            url,
            {
                "model": args.model,
                "prompt": _make_prompt_words(512, seed=9001),
                "max_tokens": 2,
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
            },
        )
        for idx, plen in enumerate([128, 512, 1024, 4096, 8192]):
            r = _post_stream_sync(
                url,
                {
                    "model": args.model,
                    "prompt": _make_prompt_words(plen, seed=5000 + idx * 17),
                    "max_tokens": 2,
                    "temperature": 0.0,
                    "ignore_eos": True,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
            )
            tps = r["prompt_tokens"] / max(r["ttft_ms"] / 1000.0, 1e-6)
            print(
                f"  target={plen:4d} | actual_p={r['prompt_tokens']:5d} | "
                f"TTFT={r['ttft_ms']:7.1f} ms | prefill={tps:7.0f} tok/s",
                flush=True,
            )


if __name__ == "__main__":
    main()
