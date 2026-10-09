# miniinfer

A small LLM inference engine for Qwen3-0.6B, built from scratch and optimized one
technique at a time. Every optimization gets a before/after number.

## Optimization ladder
| Rung | Tokens/sec | TTFT | Notes |
|------|-----------|------|-------|
| Naive (no cache) | _fill in from results/runs.jsonl_ | | |

## Workflow
    pip install -e .
    pytest -x            # parity vs Hugging Face (fp32)
    python -m miniinfer.bench.naive_baseline
