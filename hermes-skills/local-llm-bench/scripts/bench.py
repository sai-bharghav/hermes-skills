#!/usr/bin/env python3
"""Benchmark LLM endpoints: decode tok/s, prefill tok/s, TTFT.
Providers:
  ollama - Ollama native /api/chat (precise server-side token timings)
  openai - any OpenAI-compatible /chat/completions endpoint
           (vLLM, OpenAI, OpenRouter, Ollama's own /v1). Timed client-side.
Logs every run to CSV. Pure stdlib."""
import argparse, csv, json, os, sys, time, urllib.request
from datetime import datetime


def run_once_ollama(host, model, prompt, num_predict, api_key=None):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "stream": True, "options": {"num_predict": num_predict}}
    req = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    ttft = None
    final = None
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if ttft is None and obj.get("message", {}).get("content", ""):
                ttft = time.perf_counter() - start
            if obj.get("done"):
                final = obj
    wall = time.perf_counter() - start
    if not final:
        return None
    ec = final.get("eval_count", 0)
    ed = final.get("eval_duration", 0) / 1e9
    pc = final.get("prompt_eval_count", 0)
    pd = final.get("prompt_eval_duration", 0) / 1e9
    return {"decode_tps": ec / ed if ed else 0.0,
            "prefill_tps": pc / pd if pd else None,
            "ttft_ms": round(ttft * 1000) if ttft is not None else "",
            "out_tokens": ec, "prompt_tokens": pc, "wall": wall}


def run_once_openai(base_url, model, prompt, max_tokens, api_key=None):
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "stream": True, "max_tokens": max_tokens,
               "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(), headers=headers)
    start = time.perf_counter()
    ttft = first_t = last_t = None
    chunks = 0
    usage = None
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                content = (ch.get("delta") or {}).get("content")
                if content:
                    now = time.perf_counter()
                    if first_t is None:
                        first_t = now
                        ttft = now - start
                    last_t = now
                    chunks += 1
    wall = time.perf_counter() - start

    out_tokens = (usage or {}).get("completion_tokens") or chunks
    prompt_tokens = (usage or {}).get("prompt_tokens", 0)

    # Decode window: prefer first->last token span; fall back to (wall - ttft);
    # final fallback is total wall. Guard against zero/None at every step.
    if first_t is not None and last_t is not None and last_t > first_t:
        decode_window = last_t - first_t
    elif ttft is not None and wall > ttft:
        decode_window = wall - ttft
    else:
        decode_window = wall
    decode_tps = (out_tokens / decode_window) if decode_window > 0 else 0.0

    return {"decode_tps": decode_tps,
            "prefill_tps": None,  # not reported by OpenAI-compatible servers
            "ttft_ms": round(ttft * 1000) if ttft is not None else "",
            "out_tokens": out_tokens, "prompt_tokens": prompt_tokens, "wall": wall}


RUNNERS = {"ollama": run_once_ollama, "openai": run_once_openai}


def bench_model(runner, target, model, prompt, num_predict, runs,
                api_key, provider, writer):
    print(f"\n=== [{provider}] {model} ===")
    rates = []
    for i in range(1, runs + 1):
        try:
            m = runner(target, model, prompt, num_predict, api_key=api_key)
        except Exception as e:
            print(f"Run {i}: ERROR {e}", file=sys.stderr)
            return
        if not m:
            print(f"Run {i}: no metrics", file=sys.stderr)
            continue
        warmup = (i == 1 and runs > 1)
        rates.append(m["decode_tps"])
        pre = f"{m['prefill_tps']:8.1f}" if m["prefill_tps"] else "     n/a"
        ttft = str(m["ttft_ms"]) if m["ttft_ms"] != "" else "n/a"
        print(f"Run {i}: TTFT {ttft:>6} ms | "
              f"decode {m['decode_tps']:6.1f} tok/s ({m['out_tokens']} tok) | "
              f"prefill {pre} tok/s | wall {m['wall']:4.2f}s"
              + ("  [warmup]" if warmup else ""))
        writer.writerow([datetime.now().isoformat(timespec="seconds"),
                         provider, model, i, warmup,
                         round(m["decode_tps"], 1),
                         round(m["prefill_tps"], 1) if m["prefill_tps"] else "",
                         m["ttft_ms"], m["out_tokens"], m["prompt_tokens"],
                         round(m["wall"], 2)])
    if rates:
        useful = rates[1:] if len(rates) > 1 else rates
        print(f"Avg decode (excl. warmup): {sum(useful)/len(useful):.1f} tok/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemma4:latest"])
    ap.add_argument("--provider", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--host", default="http://localhost:11434",
                    help="ollama: base host | openai: base URL incl. /v1")
    ap.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY", ""),
                    help="bearer token for openai provider (or env BENCH_API_KEY)")
    ap.add_argument("--prompt",
                    default="Explain FP8 quantization for LLM inference in 3 sentences.")
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--csv", default="results.csv")
    args = ap.parse_args()

    runner = RUNNERS[args.provider]
    new_file = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "provider", "model", "run", "warmup",
                        "decode_tps", "prefill_tps", "ttft_ms",
                        "out_tokens", "prompt_tokens", "wall_s"])
        print(f"Provider: {args.provider}   Target: {args.host}")
        print(f"Logging to: {os.path.abspath(args.csv)}")
        print(f"Prompt: {args.prompt!r}   runs={args.runs} num_predict={args.num_predict}")
        for model in args.models:
            bench_model(runner, args.host, model, args.prompt, args.num_predict,
                        args.runs, args.api_key, args.provider, w)
    print(f"\nResults appended to {args.csv}")


if __name__ == "__main__":
    main()