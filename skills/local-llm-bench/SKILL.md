---
name: local-llm-bench
description: Benchmark a local Ollama model's decode tok/s, prefill tok/s, and TTFT.
version: 1.0.0
author: Sai
license: MIT
platforms: [windows, linux, macos]
metadata:
  hermes:
    tags: [benchmark, ollama, inference, local-llm]
---

# Local LLM Benchmark

Measures real single-stream inference performance of a local Ollama model:
decode throughput (tok/s), prefill throughput (tok/s), and time-to-first-token.
Uses Ollama's native /api/chat, so token counts and timings come straight from
the server and are accurate, not estimated.

## When to use this
- The user asks to benchmark, measure, or test the speed of a local model.
- The user wants tok/s, throughput, TTFT, or latency numbers for an Ollama model.
- The user wants to compare local inference performance against a cloud endpoint.

## Steps
1. Confirm Ollama is running and the target model is available. The user's default
   model is `gemma4:latest`. If unsure which models exist, run: `ollama list`
2. Run the benchmark script:
   `python ${HERMES_SKILL_DIR}/scripts/bench.py --model <model-name>`
3. Optional flags:
   - `--runs N` (default 3) — number of timed runs; run 1 is discarded as warmup.
   - `--num-predict N` (default 256) — max output tokens per run.
   - `--prompt "..."` — custom prompt.
   - `--host http://localhost:11434` — Ollama host if not the default.
4. Report the average decode throughput and note prefill tok/s and TTFT.

## Pitfalls
- Run 1 is always slower (cold-start prefill + model load) — it is excluded from
  the average on purpose. Do not report run 1 as the result.
- TTFT may show `n/a` on warm runs when Ollama returns without an incremental
  first-token chunk. This is expected and does not affect decode tok/s accuracy.
- If the script errors with a connection refused, Ollama is not running. Start it
  and confirm with `ollama ps`.

## Examples
Benchmark the default model:
`python ${HERMES_SKILL_DIR}/scripts/bench.py --model gemma4:latest`

Longer outputs, 5 runs:
`python ${HERMES_SKILL_DIR}/scripts/bench.py --model gemma4:latest --runs 5 --num-predict 512`